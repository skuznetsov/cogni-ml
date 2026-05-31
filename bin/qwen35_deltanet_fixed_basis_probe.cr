#!/usr/bin/env crystal

require "json"
require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_ffn_updown_adapter"
require "../src/ml/gguf/qwen35_mtp"
require "../src/ml/gguf/qwen35_prompt_cache"
require "../src/ml/gguf/qwen35_proposal_route"
require "../src/ml/gguf/qwen35_self_spec_updown_buffers"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL     = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_MTP       = "#{ENV["HOME"]}/.cache/cogni-ml/qwen36_mtp/Qwen3.6-27B-mtp.safetensors"
DEFAULT_TOKENIZER = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"
DEFAULT_PROMPT    = "The quick brown fox jumps over the lazy dog. Describe this scene in detail, then explain how weather, geometry, and memory interact in a compact machine learning runtime. " \
                    "Use precise technical language and include several short code-like phrases so the token stream is varied."
DEFAULT_SELF_SPEC_GPU_PIPELINE_DRAFT_BLOCK_TOKENS = 1
DEFAULT_FFN_SPARSE_BLOCK_SIZE                     = 256

module ProbeRuntime
  @@fallback_score_mode = "raw"
  @@gpu_draft_exact_refresh_interval = 0
  @@gpu_draft_exact_refresh_offsets = [] of Int32
  @@gpu_draft_exact_refresh_prefix = 0
  @@gpu_draft_update_risk_threshold : Float64? = nil
  @@gpu_draft_update_risk_layer_threshold : Float64? = nil
  @@self_spec_router_trace_io : IO? = nil
  @@self_spec_router_trace_label = "main"
  @@self_spec_branch_guard_snapshot = false
  @@self_spec_branch_guard_until_reject = false
  @@self_spec_branch_guard_overlap_next = false
  @@self_spec_branch_guard_snapshot_only_split = false
  @@self_spec_branch_guard_single_pass_checkpoint = false
  @@self_spec_branch_guard_snapshot_min_prefix = 1
  @@self_spec_branch_guard_snapshot_suffix_threshold : Float64? = nil
  @@self_spec_branch_guard_snapshot_suffix_min_threshold : Float64? = nil
  @@self_spec_branch_guard_snapshot_prefix_suffix_thresholds = [] of Tuple(Int32, Float64)
  @@self_spec_branch_guard_no_snapshot_threshold : Float64? = nil
  @@self_spec_draft_refresh_on_accept = false
  @@self_spec_draft_no_ffn_fallback_on_reject = false
  @@self_spec_draft_no_ffn_after_full_accepts = 0
  @@self_spec_draft_no_ffn_min_margin : Float64? = nil
  @@self_spec_draft_no_ffn_max_chunks : Int32? = nil
  @@self_spec_draft_updown_race_first_chunk = false
  @@self_spec_draft_updown_first_margin_threshold : Float64? = nil

  def self.fallback_score_mode : String
    @@fallback_score_mode
  end

  def self.fallback_score_mode=(mode : String)
    unless {"raw", "decayed", "update"}.includes?(mode)
      raise "unknown fallback score mode #{mode.inspect}; expected raw, decayed, or update"
    end
    @@fallback_score_mode = mode
  end

  def self.gpu_draft_exact_refresh_interval : Int32
    @@gpu_draft_exact_refresh_interval
  end

  def self.gpu_draft_exact_refresh_interval=(interval : Int32)
    raise "GPU draft exact refresh interval must be non-negative" if interval < 0
    @@gpu_draft_exact_refresh_interval = interval
  end

  def self.gpu_draft_exact_refresh_offsets : Array(Int32)
    @@gpu_draft_exact_refresh_offsets
  end

  def self.gpu_draft_exact_refresh_offsets=(offsets : Array(Int32))
    raise "GPU draft exact refresh offsets must be non-negative" if offsets.any? { |v| v < 0 }
    @@gpu_draft_exact_refresh_offsets = offsets.uniq.sort
  end

  def self.gpu_draft_exact_refresh_prefix : Int32
    @@gpu_draft_exact_refresh_prefix
  end

  def self.gpu_draft_exact_refresh_prefix=(prefix : Int32)
    raise "GPU draft exact refresh prefix must be non-negative" if prefix < 0
    @@gpu_draft_exact_refresh_prefix = prefix
  end

  def self.gpu_draft_update_risk_threshold : Float64?
    @@gpu_draft_update_risk_threshold
  end

  def self.gpu_draft_update_risk_threshold=(threshold : Float64?)
    if value = threshold
      raise "GPU draft update-risk threshold must be non-negative" if value < 0.0
    end
    @@gpu_draft_update_risk_threshold = threshold
  end

  def self.gpu_draft_update_risk_layer_threshold : Float64?
    @@gpu_draft_update_risk_layer_threshold
  end

  def self.gpu_draft_update_risk_layer_threshold=(threshold : Float64?)
    if value = threshold
      raise "GPU draft update-risk layer threshold must be non-negative" if value < 0.0
    end
    @@gpu_draft_update_risk_layer_threshold = threshold
  end

  def self.self_spec_draft_updown_race_first_chunk : Bool
    @@self_spec_draft_updown_race_first_chunk
  end

  def self.self_spec_draft_updown_race_first_chunk=(enabled : Bool)
    @@self_spec_draft_updown_race_first_chunk = enabled
  end

  def self.self_spec_draft_updown_first_margin_threshold : Float64?
    @@self_spec_draft_updown_first_margin_threshold
  end

  def self.self_spec_draft_updown_first_margin_threshold=(threshold : Float64?)
    if value = threshold
      raise "GPU pipeline pca-updown first-margin threshold must be non-negative" if value < 0.0
    end
    @@self_spec_draft_updown_first_margin_threshold = threshold
  end

  def self.self_spec_router_trace_io : IO?
    @@self_spec_router_trace_io
  end

  def self.self_spec_router_trace_io=(io : IO?)
    @@self_spec_router_trace_io = io
  end

  def self.self_spec_router_trace_label : String
    @@self_spec_router_trace_label
  end

  def self.self_spec_router_trace_label=(label : String)
    @@self_spec_router_trace_label = label
  end

  def self.self_spec_draft_refresh_on_accept : Bool
    @@self_spec_draft_refresh_on_accept
  end

  def self.self_spec_draft_refresh_on_accept=(enabled : Bool)
    @@self_spec_draft_refresh_on_accept = enabled
  end

  def self.self_spec_draft_no_ffn_fallback_on_reject : Bool
    @@self_spec_draft_no_ffn_fallback_on_reject
  end

  def self.self_spec_draft_no_ffn_fallback_on_reject=(enabled : Bool)
    @@self_spec_draft_no_ffn_fallback_on_reject = enabled
  end

  def self.self_spec_draft_no_ffn_after_full_accepts : Int32
    @@self_spec_draft_no_ffn_after_full_accepts
  end

  def self.self_spec_draft_no_ffn_after_full_accepts=(chunks : Int32)
    raise "self-spec no-FFN after-full-accepts must be non-negative" if chunks < 0
    @@self_spec_draft_no_ffn_after_full_accepts = chunks
  end

  def self.self_spec_draft_no_ffn_min_margin : Float64?
    @@self_spec_draft_no_ffn_min_margin
  end

  def self.self_spec_draft_no_ffn_min_margin=(margin : Float64?)
    if value = margin
      raise "self-spec no-FFN min-margin must be non-negative" if value < 0.0
    end
    @@self_spec_draft_no_ffn_min_margin = margin
  end

  def self.self_spec_draft_no_ffn_max_chunks : Int32?
    @@self_spec_draft_no_ffn_max_chunks
  end

  def self.self_spec_draft_no_ffn_max_chunks=(chunks : Int32?)
    if value = chunks
      raise "self-spec no-FFN max chunks must be non-negative" if value < 0
    end
    @@self_spec_draft_no_ffn_max_chunks = chunks
  end

  def self.self_spec_branch_guard_snapshot : Bool
    @@self_spec_branch_guard_snapshot
  end

  def self.self_spec_branch_guard_snapshot=(enabled : Bool)
    @@self_spec_branch_guard_snapshot = enabled
  end

  def self.self_spec_branch_guard_until_reject : Bool
    @@self_spec_branch_guard_until_reject
  end

  def self.self_spec_branch_guard_until_reject=(enabled : Bool)
    @@self_spec_branch_guard_until_reject = enabled
  end

  def self.self_spec_branch_guard_overlap_next : Bool
    @@self_spec_branch_guard_overlap_next
  end

  def self.self_spec_branch_guard_overlap_next=(enabled : Bool)
    @@self_spec_branch_guard_overlap_next = enabled
  end

  def self.self_spec_branch_guard_snapshot_only_split : Bool
    @@self_spec_branch_guard_snapshot_only_split
  end

  def self.self_spec_branch_guard_snapshot_only_split=(enabled : Bool)
    @@self_spec_branch_guard_snapshot_only_split = enabled
  end

  def self.self_spec_branch_guard_single_pass_checkpoint : Bool
    @@self_spec_branch_guard_single_pass_checkpoint
  end

  def self.self_spec_branch_guard_single_pass_checkpoint=(enabled : Bool)
    @@self_spec_branch_guard_single_pass_checkpoint = enabled
  end

  def self.self_spec_branch_guard_snapshot_min_prefix : Int32
    @@self_spec_branch_guard_snapshot_min_prefix
  end

  def self.self_spec_branch_guard_snapshot_min_prefix=(value : Int32)
    raise "branch guard snapshot min prefix must be positive" if value <= 0
    @@self_spec_branch_guard_snapshot_min_prefix = value
  end

  def self.self_spec_branch_guard_snapshot_suffix_threshold : Float64?
    @@self_spec_branch_guard_snapshot_suffix_threshold
  end

  def self.self_spec_branch_guard_snapshot_suffix_threshold=(threshold : Float64?)
    if value = threshold
      raise "branch guard snapshot suffix threshold must be non-negative" if value < 0.0
    end
    @@self_spec_branch_guard_snapshot_suffix_threshold = threshold
  end

  def self.self_spec_branch_guard_snapshot_suffix_min_threshold : Float64?
    @@self_spec_branch_guard_snapshot_suffix_min_threshold
  end

  def self.self_spec_branch_guard_snapshot_suffix_min_threshold=(threshold : Float64?)
    if value = threshold
      raise "branch guard snapshot suffix min threshold must be non-negative" if value < 0.0
    end
    @@self_spec_branch_guard_snapshot_suffix_min_threshold = threshold
  end

  def self.self_spec_branch_guard_snapshot_prefix_suffix_thresholds : Array(Tuple(Int32, Float64))
    @@self_spec_branch_guard_snapshot_prefix_suffix_thresholds
  end

  def self.self_spec_branch_guard_snapshot_prefix_suffix_thresholds=(thresholds : Array(Tuple(Int32, Float64)))
    thresholds.each do |min_prefix, threshold|
      raise "branch guard snapshot prefix threshold min-prefix must be positive" if min_prefix <= 0
      raise "branch guard snapshot prefix threshold must be non-negative" if threshold < 0.0
    end
    @@self_spec_branch_guard_snapshot_prefix_suffix_thresholds = thresholds.sort_by { |pair| pair[0] }
  end

  def self.self_spec_branch_guard_no_snapshot_threshold : Float64?
    @@self_spec_branch_guard_no_snapshot_threshold
  end

  def self.self_spec_branch_guard_no_snapshot_threshold=(threshold : Float64?)
    if value = threshold
      raise "branch guard no-snapshot threshold must be non-negative" if value < 0.0
    end
    @@self_spec_branch_guard_no_snapshot_threshold = threshold
  end
end

private alias BasisSet = Array(Array(Array(Float64)))
private alias LayerVectorMap = Hash(Int32, BasisSet)
private alias LayerBasisMap = Hash(Int32, BasisSet)
private alias FFNBasisMap = Hash(Int32, Array(Array(Float64)))
private alias FFNActivationSample = NamedTuple(ffn_in: Array(Float64), activation: Array(Float64))
private alias FFNBlockSparsityLayerStats = NamedTuple(layer: Int32, vectors: Int32, dim: Int32, block_size: Int32, blocks: Int32, read90_mean: Float64, read95_mean: Float64, read99_mean: Float64)
private alias BlockResidualSample = NamedTuple(inp: Array(Float64), out: Array(Float64), delta: Array(Float64))
private alias LayerBlock = NamedTuple(start: Int32, end: Int32)
private alias NamedPrompt = NamedTuple(name: String, text: String)
private alias PromptTokenSet = NamedTuple(name: String, token_ids: Array(Int32))
private alias BlockSurrogateSuiteRow = NamedTuple(prompt: String, block: String, mode: String, rank: Int32, gamma: Int32, parity: Bool, verifier_parity: Bool, accept_rate: Float64, rejections: Int32, accepted_draft_tokens: Int32, proposed_tokens: Int32, chunks: Int32, full_accept_chunks: Int32, correction_steps: Int32, draft_top2_hit_rate: Float64, draft_top5_hit_rate: Float64, draft_margin_min: Float64, baseline_decode_ms: Float64, draft_ms: Float64, verifier_ms: Float64, self_seq_decode_ms: Float64, ideal_overlap_decode_ms: Float64, cpu_seq_speedup: Float64, ideal_overlap_speedup: Float64, hidden_cos_mean: Float64, hidden_cos_min: Float64, rel_rmse: Float64, delta_rel_rmse: Float64)
private alias BlockSurrogateTreeSuiteRow = NamedTuple(prompt: String, block: String, mode: String, rank: Int32, gamma: Int32, top_k: Int32, prefill_seed: Bool, branch_verify: Bool, select_advance: Bool, warmup_tokens: Int32, prefill_seed_tokens: Int32, tree_tokens: Int32, parity: Bool, full_rescue_chunks: Int32, chunks: Int32, misses: Int32, draft_steps: Int32, top1_rate: Float64, topk_rate: Float64, avg_rank_branch_tokens: Float64, avg_full_branch_tokens: Float64, avg_rank_branch_tokens_total: Float64, avg_full_branch_tokens_total: Float64, branch_tokens_rank: Int32, branch_tokens_full: Int32, branch_verify_attempts: Int32, branch_verify_wasted_attempts: Int32, branch_verify_corrections: Int32, branch_verify_ms: Float64, branch_verify_fork_ms: Float64, branch_verify_forward_ms: Float64, correction_steps: Int32, hidden_cos_mean: Float64, rel_rmse: Float64)
private alias HybridRoute = NamedTuple(name: String, noffn: Set(Int32)?, updown: Set(Int32)?)
private alias RouteScoreRow = NamedTuple(prompt: String, mode: String, split: String, route: String, updown_rank: Int32?, parity: Bool, accept_rate: Float64, rejections: Int32, plain_speedup: Float64, overlap_ms: Float64, plain_exact_ms: Float64, draft_wait_ms: Float64, replay_ms: Float64, tree2_margin_min: Float64, tree2_reject_margin_min: Float64, residual_mean: Float64?, residual_p90: Float64?, residual_max: Float64?, repeat_rate: Float64?, bigram_repeat_rate: Float64?, unique_rate: Float64?)
private alias DraftBodyScoreRow = NamedTuple(prompt: String, mode: String, split: String, body: String, updown_rank: Int32?, parity: Bool, accept_rate: Float64, rejections: Int32, draft_updown_chunks: Int32, plain_speedup: Float64, overlap_ms: Float64, plain_exact_ms: Float64, draft_next_ms: Float64, verifier_ms: Float64, draft_wait_ms: Float64, replay_ms: Float64)
private alias RiskOfframpScoreRow = NamedTuple(prompt: String, mode: String, split: String, threshold: String, parity: Bool, accept_rate: Float64, rejections: Int32, plain_speedup: Float64, overlap_ms: Float64, plain_exact_ms: Float64, draft_wait_ms: Float64, replay_ms: Float64, risk_hits: Int32, delayed_blocks: Int32, delayed_tokens: Int32, margin_min: Float64, reject_margin_min: Float64)
private alias MtpSelfDraftFusionRow = NamedTuple(index: Int32, exact: Int32, self_id: Int32, self_second_id: Int32, mtp_rank: Int32, self_hit: Bool, self_top2_hit: Bool, mtp_hit: Bool, mtp_k2_hit: Bool, union_hit: Bool, union_k2_hit: Bool, agreement: Bool, union_size: Int32, union_k2_size: Int32, mtp_first_attempts: Int32, self_first_attempts: Int32)

private def fnv1a64_hex(bytes : Bytes) : String
  hash = 0xcbf29ce484222325_u64
  bytes.each do |b|
    hash = (hash ^ b.to_u64) &* 0x100000001b3_u64
  end
  hash.to_s(16)
end

private def safe_prompt_label(name : String, fallback : String) : String
  safe = name.gsub(/[^A-Za-z0-9_.-]/, "_")
  safe.empty? ? fallback : safe
end

private def probe_prompt_category(name : String) : String
  head = name.split('_', 2)[0]? || ""
  case head
  when "", "prompt", "main"
    "unknown"
  when "templ"
    "template"
  else
    head
  end
end

private def prompt_category_allowed?(prompt_name : String, allowed : Array(String)) : Bool
  return true if allowed.empty?
  category = probe_prompt_category(prompt_name)
  allowed.includes?(category) || allowed.includes?(prompt_name)
end

private struct RecurrentSample
  getter inp : Array(Float32)
  getter q : Array(Float32)
  getter k : Array(Float32)
  getter v : Array(Float32)
  getter ghead : Array(Float32)
  getter beta : Array(Float32)
  getter z : Array(Float32)

  def initialize(@q, @k, @v, @ghead, @beta, @z = [] of Float32, @inp = [] of Float32)
  end
end

private class LowRankState
  property initialized : Bool = false
  property full_state_current : Bool = true
  property m : Array(Float32) = [] of Float32
  property m_buf : ML::MetalBuffer?
  property basis_buf : ML::MetalBuffer?
  property basis_key : String = ""
  property updown_x_mean_buf : ML::MetalBuffer?
  property updown_c_mean_buf : ML::MetalBuffer?
  property updown_coeff_w_buf : ML::MetalBuffer?
  property updown_down_buf : ML::MetalBuffer?
  property updown_key : String = ""
  property approx_steps : Int32 = 0
  property fallback_steps : Int32 = 0
end

private class GpuDraftBlock
  getter submissions : Array(ML::GGUF::Qwen35Metal::DecodeWaveSubmission)
  getter state : ML::GGUF::Qwen35CPU::State
  getter lr_bufs : Hash(Int32, ML::MetalBuffer)
  getter full_current : Hash(Int32, Bool)
  getter use_updown : Bool
  getter use_noffn : Bool

  def initialize(@submissions, @state, @lr_bufs, @full_current, @use_updown, @use_noffn)
  end
end

private struct FFNAdapter
  getter basis : Array(Array(Float64))
  getter down_basis : Array(Array(Float32))

  def initialize(@basis : Array(Array(Float64)), @down_basis : Array(Array(Float32)))
  end
end

private alias FFNAdapterMap = Hash(Int32, FFNAdapter)

private alias FFNUpDownAdapter = ML::GGUF::Qwen35FFNUpDownAdapter
private alias FFNUpDownAdapterMap = ML::GGUF::Qwen35FFNUpDownAdapterMap

private struct FFNBlockSelector
  getter samples : Array(FFNActivationSample)
  getter blocks_by_percent : Hash(Int32, Array(Set(Int32)))
  getter block_size : Int32

  def initialize(@samples : Array(FFNActivationSample),
                 @blocks_by_percent : Hash(Int32, Array(Set(Int32))),
                 @block_size : Int32)
  end
end

private alias FFNBlockSelectorMap = Hash(Int32, FFNBlockSelector)

private struct BlockResidualSurrogate
  getter block_start : Int32
  getter block_end : Int32
  getter x_mean : Array(Float64)
  getter delta_mean : Array(Float64)
  getter input_basis : Array(Array(Float64))
  getter delta_basis : Array(Array(Float64))
  getter coeff_weights : Array(Array(Float64))

  def initialize(@block_start : Int32,
                 @block_end : Int32,
                 @x_mean : Array(Float64),
                 @delta_mean : Array(Float64),
                 @input_basis : Array(Array(Float64)),
                 @delta_basis : Array(Array(Float64)),
                 @coeff_weights : Array(Array(Float64)))
  end
end

private struct BlockResidualMixture
  getter centroids : Array(Array(Float64))
  getter adapters : Array(BlockResidualSurrogate)
  getter cluster_sizes : Array(Int32)
  getter global_adapter : BlockResidualSurrogate
  getter feature_mean : Array(Float64)
  getter feature_basis : Array(Array(Float64))

  def initialize(@centroids : Array(Array(Float64)),
                 @adapters : Array(BlockResidualSurrogate),
                 @cluster_sizes : Array(Int32),
                 @global_adapter : BlockResidualSurrogate,
                 @feature_mean : Array(Float64),
                 @feature_basis : Array(Array(Float64)))
  end
end

private class WbaTrace
  @base : Time::Instant
  @events = [] of String
  @mutex = Mutex.new

  def initialize(@label : String)
    @base = Time.instant
  end

  def self.enabled? : Bool
    ENV["QWEN35_WBA"]? == "1"
  end

  def self.maybe(label : String) : WbaTrace?
    enabled? ? new(label) : nil
  end

  def mark(lane : String, stage : String, t0 : Time::Instant, t1 : Time::Instant) : Nil
    start_ms = (t0 - @base).total_milliseconds
    end_ms = (t1 - @base).total_milliseconds
    dur_ms = (t1 - t0).total_milliseconds
    event = sprintf("wba label=%s lane=%s stage=%s start_ms=%.3f end_ms=%.3f dur_ms=%.3f",
      @label, lane, stage, start_ms, end_ms, dur_ms)
    @mutex.synchronize { @events << event }
  end

  def point(lane : String, stage : String, t : Time::Instant) : Nil
    ms = (t - @base).total_milliseconds
    event = sprintf("wba label=%s lane=%s stage=%s at_ms=%.3f",
      @label, lane, stage, ms)
    @mutex.synchronize { @events << event }
  end

  def flush : Nil
    events = @mutex.synchronize { @events.dup }
    return if events.empty?
    events.each { |event| STDERR.puts event }
  end
end

private def softplus(x : Float32) : Float32
  x > 20.0_f32 ? x : Math.log(1.0_f32 + Math.exp(x)).to_f32
end

private def silu!(x : Array(Float32)) : Nil
  x.size.times do |i|
    v = x[i]
    x[i] = v / (1.0_f32 + Math.exp(-v).to_f32)
  end
end

private def l2_norm_slice!(x : Array(Float32), offset : Int32, len : Int32, eps : Float32) : Nil
  ss = 0.0_f64
  len.times { |i| ss += x[offset + i].to_f64 * x[offset + i].to_f64 }
  inv = (1.0 / Math.sqrt(ss + eps.to_f64)).to_f32
  len.times { |i| x[offset + i] *= inv }
end

private def recurrent_k_vectors_for_prompt(weights : ML::GGUF::Qwen35Weights,
                                           token_ids : Array(Int32),
                                           layer_index : Int32) : Array(Array(Array(Float64)))
  hp = weights.hparams
  target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) ||
                 raise "layer #{layer_index} is not recurrent"
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  qkv_dim = 2 * h_k * s + h_v * s
  conv_k = hp.ssm_conv_kernel
  conv_state = Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: token_ids.size + 2)
  per_head = Array.new(h_k) { [] of Array(Float64) }

  token_ids.each_with_index do |token_id, pos|
    x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)

    layer_index.times do |il|
      case layer = weights.layers[il]
      in ML::GGUF::Qwen35FullAttnWeights
        x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      in ML::GGUF::Qwen35RecurrentWeights
        x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      end
    end

    cur = ML::GGUF::Qwen35CPU.rms_norm(x, target_layer.attn_norm, hp.rms_eps)
    qkv_mixed = ML::GGUF::Qwen35CPU.qmatvec_nobias(target_layer.attn_qkv_qw, cur)

    conv_out = Array(Float32).new(qkv_dim) do |ch|
      acc = 0.0_f32
      w_base = ch * conv_k
      (conv_k - 1).times do |t|
        acc += conv_state[t * qkv_dim + ch] * target_layer.ssm_conv1d[w_base + t]
      end
      acc + qkv_mixed[ch] * target_layer.ssm_conv1d[w_base + (conv_k - 1)]
    end

    (conv_k - 2).times do |t|
      src = (t + 1) * qkv_dim
      dst = t * qkv_dim
      qkv_dim.times { |ch| conv_state[dst + ch] = conv_state[src + ch] }
    end
    last = (conv_k - 2) * qkv_dim
    qkv_dim.times { |ch| conv_state[last + ch] = qkv_mixed[ch] }

    silu!(conv_out)
    k_offset = h_k * s
    h_k.times { |h| l2_norm_slice!(conv_out, k_offset + h * s, s, hp.rms_eps) }

    h_k.times do |h|
      off = k_offset + h * s
      per_head[h] << Array.new(s) { |d| conv_out[off + d].to_f64 }
    end
  end

  per_head
end

private def recurrent_samples_for_prompt(weights : ML::GGUF::Qwen35Weights,
                                         token_ids : Array(Int32),
                                         layer_index : Int32) : Array(RecurrentSample)
  hp = weights.hparams
  target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) ||
                 raise "layer #{layer_index} is not recurrent"
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  qkv_dim = 2 * h_k * s + h_v * s
  conv_k = hp.ssm_conv_kernel
  conv_state = Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: token_ids.size + 2)
  samples = [] of RecurrentSample

  token_ids.each_with_index do |token_id, pos|
    x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)

    layer_index.times do |il|
      case layer = weights.layers[il]
      in ML::GGUF::Qwen35FullAttnWeights
        x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      in ML::GGUF::Qwen35RecurrentWeights
        x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      end
    end

    cur = ML::GGUF::Qwen35CPU.rms_norm(x, target_layer.attn_norm, hp.rms_eps)
    proj = ML::GGUF::Qwen35CPU.qmatvec_many([target_layer.attn_qkv_qw, target_layer.attn_gate_qw, target_layer.ssm_alpha_qw, target_layer.ssm_beta_qw], cur)
    qkv_mixed = proj[0]
    z = proj[1]
    alpha = proj[2]
    beta = proj[3]
    h_v.times { |i| beta[i] = 1.0_f32 / (1.0_f32 + Math.exp(-beta[i]).to_f32) }
    ghead = Array(Float32).new(h_v) do |i|
      Math.exp((softplus(alpha[i] + target_layer.ssm_dt_bias[i]) * target_layer.ssm_a[i]).to_f64).to_f32
    end

    conv_out = Array(Float32).new(qkv_dim) do |ch|
      acc = 0.0_f32
      w_base = ch * conv_k
      (conv_k - 1).times do |t|
        acc += conv_state[t * qkv_dim + ch] * target_layer.ssm_conv1d[w_base + t]
      end
      acc + qkv_mixed[ch] * target_layer.ssm_conv1d[w_base + (conv_k - 1)]
    end

    (conv_k - 2).times do |t|
      src = (t + 1) * qkv_dim
      dst = t * qkv_dim
      qkv_dim.times { |ch| conv_state[dst + ch] = conv_state[src + ch] }
    end
    last = (conv_k - 2) * qkv_dim
    qkv_dim.times { |ch| conv_state[last + ch] = qkv_mixed[ch] }

    silu!(conv_out)
    q_conv = Array(Float32).new(h_k * s) { |i| conv_out[i] }
    k_conv = Array(Float32).new(h_k * s) { |i| conv_out[h_k * s + i] }
    v_conv = Array(Float32).new(h_v * s) { |i| conv_out[2 * h_k * s + i] }
    h_k.times do |h|
      l2_norm_slice!(q_conv, h * s, s, hp.rms_eps)
      l2_norm_slice!(k_conv, h * s, s, hp.rms_eps)
    end
    samples << RecurrentSample.new(q_conv, k_conv, v_conv, ghead, beta, z, x.dup)
  end

  samples
end

private def dot(a : Array(Float64), b : Array(Float64)) : Float64
  acc = 0.0
  a.size.times { |i| acc += a[i] * b[i] }
  acc
end

private def residual_norm(v : Array(Float64), basis : Array(Array(Float64)), rank : Int32) : Float64
  residual = v.dup
  limit = Math.min(rank, basis.size)
  limit.times do |i|
    b = basis[i]
    coeff = dot(residual, b)
    residual.size.times { |d| residual[d] -= coeff * b[d] }
  end
  Math.sqrt(dot(residual, residual))
end

private def greedy_basis(vectors : Array(Array(Float64)), max_rank : Int32, eps : Float64 = 1.0e-6) : Array(Array(Float64))
  basis = [] of Array(Float64)
  vectors.each do |v|
    break if basis.size >= max_rank
    residual = v.dup
    basis.each do |b|
      coeff = dot(residual, b)
      residual.size.times { |d| residual[d] -= coeff * b[d] }
    end
    norm = Math.sqrt(dot(residual, residual))
    next if norm <= eps
    basis << residual.map { |x| x / norm }
  end
  basis
end

private def norm(v : Array(Float64)) : Float64
  Math.sqrt(dot(v, v))
end

private def orthogonalize!(v : Array(Float64), basis : Array(Array(Float64))) : Nil
  basis.each do |b|
    coeff = dot(v, b)
    v.size.times { |i| v[i] -= coeff * b[i] }
  end
end

private def covariance_matvec(vectors : Array(Array(Float64)), x : Array(Float64)) : Array(Float64)
  out = Array.new(x.size, 0.0)
  vectors.each do |sample|
    coeff = dot(sample, x)
    x.size.times { |i| out[i] += sample[i] * coeff }
  end
  out
end

private def pca_basis(vectors : Array(Array(Float64)), max_rank : Int32,
                      iters : Int32 = 24, eps : Float64 = 1.0e-7) : Array(Array(Float64))
  return [] of Array(Float64) if vectors.empty?

  dim = vectors[0].size
  basis = [] of Array(Float64)
  max_rank.times do |rank|
    # Deterministic non-random start vector. The sinusoid avoids selecting the
    # same axis repeatedly when the covariance spectrum has near ties.
    x = Array.new(dim) { |i| Math.sin((rank + 1) * (i + 1) * 0.0137) + Math.cos((rank + 3) * (i + 1) * 0.0071) }
    orthogonalize!(x, basis)
    n = norm(x)
    break if n <= eps
    dim.times { |i| x[i] /= n }

    iters.times do
      y = covariance_matvec(vectors, x)
      orthogonalize!(y, basis)
      yn = norm(y)
      break if yn <= eps
      dim.times { |i| x[i] = y[i] / yn }
    end

    y = covariance_matvec(vectors, x)
    orthogonalize!(y, basis)
    lambda = dot(x, y)
    break if lambda.abs <= eps

    # Re-normalize after the final orthogonalization step to keep residual
    # measurements comparable to greedy MGS.
    orthogonalize!(x, basis)
    xn = norm(x)
    break if xn <= eps
    basis << x.map { |v| v / xn }
  end

  basis
end

private def append_orthonormal_basis!(basis : Array(Array(Float64)),
                                      candidate : Array(Float64),
                                      eps : Float64 = 1.0e-7) : Bool
  v = candidate.dup
  orthogonalize!(v, basis)
  n = norm(v)
  return false if n <= eps
  basis << v.map { |x| x / n }
  true
end

private def interleaved_basis(primary : Array(Array(Float64)),
                              secondary : Array(Array(Float64)),
                              max_rank : Int32) : Array(Array(Float64))
  basis = [] of Array(Float64)
  max_size = Math.max(primary.size, secondary.size)
  max_size.times do |i|
    append_orthonormal_basis!(basis, primary[i]) if i < primary.size && basis.size < max_rank
    append_orthonormal_basis!(basis, secondary[i]) if i < secondary.size && basis.size < max_rank
    break if basis.size >= max_rank
  end
  basis
end

private def build_basis(vectors : Array(Array(Float64)), max_rank : Int32,
                        mode : String, pca_iters : Int32) : Array(Array(Float64))
  case mode
  when "greedy"
    greedy_basis(vectors, max_rank)
  when "pca"
    pca_basis(vectors, max_rank, pca_iters)
  else
    raise "unsupported basis mode #{mode.inspect}; expected greedy or pca"
  end
end

private def basis_rank_range(bases : BasisSet) : NamedTuple(min: Int32, max: Int32)
  sizes = bases.map(&.size)
  {min: sizes.min, max: sizes.max}
end

private def basis_rank_note(bases : BasisSet, requested_rank : Int32) : String
  range = basis_rank_range(bases)
  note = "effective_basis_rank=#{range[:min]}..#{range[:max]}"
  if requested_rank > range[:min]
    note += " requested_rank=#{requested_rank} note=requested_rank_exceeds_some_effective_bases"
  end
  note
end

private def route_residual_stats(layer_vectors : LayerVectorMap,
                                 layer_bases : LayerBasisMap,
                                 rank : Int32,
                                 calib_count : Int32,
                                 thresholds : Array(Float64))
  residuals = [] of Float64
  layer_vectors.keys.sort.each do |il|
    vectors = layer_vectors[il]
    bases = layer_bases[il]
    vectors.each_with_index do |head_vectors, head|
      next if calib_count >= head_vectors.size
      head_vectors[calib_count, head_vectors.size - calib_count].each do |v|
        residuals << residual_norm(v, bases[head], rank)
      end
    end
  end
  raise "route residual stats require held-out vectors" if residuals.empty?

  sorted = residuals.sort
  mean = residuals.sum / residuals.size
  pass_rates = thresholds.map do |threshold|
    passed = residuals.count { |r| r <= threshold }
    {threshold: threshold, rate: 100.0 * passed / residuals.size}
  end
  {
    count:      residuals.size,
    mean:       mean,
    p50:        sorted[sorted.size // 2],
    p90:        sorted[(sorted.size * 90 // 100).clamp(0, sorted.size - 1)],
    p99:        sorted[(sorted.size * 99 // 100).clamp(0, sorted.size - 1)],
    max:        sorted[-1],
    pass_rates: pass_rates,
  }
end

private def prompt_route_feature_note(name : String,
                                      layer_ids : Array(Int32),
                                      rank : Int32,
                                      token_count : Int32,
                                      calib_count : Int32,
                                      layer_vectors : LayerVectorMap,
                                      layer_bases : LayerBasisMap,
                                      thresholds : Array(Float64)) : String
  stats = route_residual_stats(layer_vectors, layer_bases, rank, calib_count, thresholds)
  pass = stats[:pass_rates].map do |entry|
    "#{entry[:threshold].round(4)}:#{entry[:rate].round(2)}%"
  end
  "self_spec_prompt_route_features name=#{name} layers=#{layer_ids.join(',')} rank=#{rank} token_vectors=#{token_count} calib_tokens=#{calib_count} heldout_tokens=#{token_count - calib_count} residual_count=#{stats[:count]} residual_mean=#{stats[:mean].round(6)} residual_p50=#{stats[:p50].round(6)} residual_p90=#{stats[:p90].round(6)} residual_p99=#{stats[:p99].round(6)} residual_max=#{stats[:max].round(6)} pass_rates=#{pass.join(',')}"
end

private def prompt_route_layer_feature_notes(name : String,
                                             layer_ids : Array(Int32),
                                             rank : Int32,
                                             token_count : Int32,
                                             calib_count : Int32,
                                             layer_vectors : LayerVectorMap,
                                             layer_bases : LayerBasisMap,
                                             thresholds : Array(Float64)) : Array(String)
  layer_ids.map do |il|
    single_vectors = {} of Int32 => BasisSet
    single_bases = {} of Int32 => BasisSet
    single_vectors[il] = layer_vectors[il]
    single_bases[il] = layer_bases[il]
    stats = route_residual_stats(single_vectors, single_bases, rank, calib_count, thresholds)
    pass = stats[:pass_rates].map do |entry|
      "#{entry[:threshold].round(4)}:#{entry[:rate].round(2)}%"
    end
    "self_spec_prompt_route_layer_features name=#{name} layer=#{il} rank=#{rank} token_vectors=#{token_count} calib_tokens=#{calib_count} heldout_tokens=#{token_count - calib_count} residual_count=#{stats[:count]} residual_mean=#{stats[:mean].round(6)} residual_p50=#{stats[:p50].round(6)} residual_p90=#{stats[:p90].round(6)} residual_p99=#{stats[:p99].round(6)} residual_max=#{stats[:max].round(6)} pass_rates=#{pass.join(',')}"
  end
end

private def self_spec_residual_router_thresholds(thresholds : Array(Float64),
                                                 pass_threshold : Float64?) : Array(Float64)
  values = thresholds.dup
  if threshold = pass_threshold
    values << threshold unless values.any? { |value| (value - threshold).abs < 1e-9 }
  end
  values
end

private def self_spec_residual_router_decision(stats,
                                               mean_max : Float64?,
                                               pass_threshold : Float64?,
                                               pass_rate_min : Float64?)
  enabled = !mean_max.nil? || !pass_threshold.nil? || !pass_rate_min.nil?
  pass_rate = 0.0
  if threshold = pass_threshold
    if entry = stats[:pass_rates].find { |row| (row[:threshold] - threshold).abs < 1e-9 }
      pass_rate = entry[:rate]
    end
  end

  run = true
  reasons = [] of String
  if limit = mean_max
    if stats[:mean] > limit
      run = false
      reasons << "mean"
    end
  end
  if min_rate = pass_rate_min
    if pass_rate < min_rate
      run = false
      reasons << "pass_rate"
    end
  end

  {
    enabled:   enabled,
    run:       run,
    reason:    run ? "run" : reasons.join("+"),
    pass_rate: pass_rate,
  }
end

private def self_spec_residual_router_note(decision,
                                           stats,
                                           mean_max : Float64?,
                                           pass_threshold : Float64?,
                                           pass_rate_min : Float64?) : String
  return "" unless decision[:enabled]

  mean_limit = mean_max.nil? ? "none" : mean_max.not_nil!.round(6).to_s
  pass_threshold_label = pass_threshold.nil? ? "none" : pass_threshold.not_nil!.round(4).to_s
  pass_rate_min_label = pass_rate_min.nil? ? "none" : pass_rate_min.not_nil!.round(2).to_s
  " residual_router=#{decision[:run] ? "run" : "skip"} residual_router_reason=#{decision[:reason]} residual_mean=#{stats[:mean].round(6)} residual_p50=#{stats[:p50].round(6)} residual_p90=#{stats[:p90].round(6)} residual_p99=#{stats[:p99].round(6)} residual_mean_max=#{mean_limit} residual_pass_threshold=#{pass_threshold_label} residual_pass_rate=#{decision[:pass_rate].round(2)} residual_pass_rate_min=#{pass_rate_min_label}"
end

private def self_spec_prompt_value_stats(token_ids : Array(Int32),
                                         calib_count : Int32)
  start = calib_count.clamp(0, token_ids.size)
  span = token_ids[start, token_ids.size - start]
  if span.empty?
    return {
      count:                    0,
      unique_count:             0,
      unique_rate:              100.0,
      repeat_rate:              0.0,
      bigram_count:             0,
      bigram_unique_count:      0,
      bigram_repeat_rate:       0.0,
      adjacent_repeat_rate:     0.0,
    }
  end

  counts = Hash(Int32, Int32).new(0)
  span.each { |id| counts[id] += 1 }
  repeated = 0
  counts.each_value do |count|
    repeated += count - 1 if count > 1
  end

  adjacent_repeats = 0
  bigram_counts = Hash(Tuple(Int32, Int32), Int32).new(0)
  if span.size > 1
    (span.size - 1).times do |i|
      adjacent_repeats += 1 if span[i] == span[i + 1]
      bigram_counts[{span[i], span[i + 1]}] += 1
    end
  end
  repeated_bigrams = 0
  bigram_counts.each_value do |count|
    repeated_bigrams += count - 1 if count > 1
  end
  bigram_total = Math.max(span.size - 1, 0)

  {
    count:                    span.size,
    unique_count:             counts.size,
    unique_rate:              100.0 * counts.size / span.size,
    repeat_rate:              100.0 * repeated / span.size,
    bigram_count:             bigram_total,
    bigram_unique_count:      bigram_counts.size,
    bigram_repeat_rate:       bigram_total > 0 ? (100.0 * repeated_bigrams / bigram_total) : 0.0,
    adjacent_repeat_rate:     bigram_total > 0 ? (100.0 * adjacent_repeats / bigram_total) : 0.0,
  }
end

private def self_spec_value_router_decision(stats,
                                            repeat_rate_min : Float64?,
                                            bigram_repeat_rate_min : Float64?,
                                            unique_rate_max : Float64?)
  enabled = !repeat_rate_min.nil? || !bigram_repeat_rate_min.nil? || !unique_rate_max.nil?
  run = true
  reasons = [] of String
  if limit = repeat_rate_min
    if stats[:repeat_rate] < limit
      run = false
      reasons << "repeat_rate"
    end
  end
  if limit = bigram_repeat_rate_min
    if stats[:bigram_repeat_rate] < limit
      run = false
      reasons << "bigram_repeat_rate"
    end
  end
  if limit = unique_rate_max
    if stats[:unique_rate] > limit
      run = false
      reasons << "unique_rate"
    end
  end

  {
    enabled: enabled,
    run:     run,
    reason:  run ? "run" : reasons.join("+"),
  }
end

private def self_spec_value_router_note(decision,
                                        stats,
                                        repeat_rate_min : Float64?,
                                        bigram_repeat_rate_min : Float64?,
                                        unique_rate_max : Float64?) : String
  return "" unless decision[:enabled]

  repeat_min_label = repeat_rate_min.nil? ? "none" : repeat_rate_min.not_nil!.round(2).to_s
  bigram_min_label = bigram_repeat_rate_min.nil? ? "none" : bigram_repeat_rate_min.not_nil!.round(2).to_s
  unique_max_label = unique_rate_max.nil? ? "none" : unique_rate_max.not_nil!.round(2).to_s
  " value_router=#{decision[:run] ? "run" : "skip"} value_router_reason=#{decision[:reason]} value_token_count=#{stats[:count]} value_unique_rate=#{stats[:unique_rate].round(2)} value_repeat_rate=#{stats[:repeat_rate].round(2)} value_bigram_repeat_rate=#{stats[:bigram_repeat_rate].round(2)} value_adjacent_repeat_rate=#{stats[:adjacent_repeat_rate].round(2)} value_repeat_rate_min=#{repeat_min_label} value_bigram_repeat_rate_min=#{bigram_min_label} value_unique_rate_max=#{unique_max_label}"
end

private def sorted_percentile(values : Array(Float64), p : Float64) : Float64
  return 0.0 if values.empty?
  sorted = values.sort
  idx = ((sorted.size - 1) * p).round.to_i.clamp(0, sorted.size - 1)
  sorted[idx]
end

private def decay_tau(g : Float64) : Float64?
  return nil unless g > 0.0 && g < 1.0
  -1.0 / Math.log(g)
end

private def dn_regime_feature_notes(name : String,
                                    layer_index : Int32,
                                    rank : Int32,
                                    token_count : Int32,
                                    calib_count : Int32,
                                    samples : Array(RecurrentSample),
                                    bases : BasisSet,
                                    residual_thresholds : Array(Float64),
                                    g_cuts : Array(Float64)) : Array(String)
  raise "DN regime features require recurrent samples" if samples.empty?
  raise "DN regime features require held-out samples" unless calib_count < samples.size

  h_k = bases.size
  s = bases[0][0].size
  h_v = samples[0].ghead.size
  heldout = samples[calib_count, samples.size - calib_count]
  values = [] of NamedTuple(head: Int32, g: Float64, tau: Float64?, beta: Float64, residual: Float64, decayed_residual: Float64, update_residual: Float64)

  heldout.each do |sample|
    h_v.times do |h|
      k_head = h % h_k
      g = sample.ghead[h].to_f64
      beta = sample.beta[h].to_f64
      residual = residual_norm_f32(sample.k, k_head * s, bases[k_head], rank)
      values << {
        head:             h,
        g:                g,
        tau:              decay_tau(g),
        beta:             beta,
        residual:         residual,
        decayed_residual: g * residual,
        update_residual:  beta * residual,
      }
    end
  end

  format_stats = ->(prefix : String, rows : Array(typeof(values[0]))) {
    gs = rows.map { |r| r[:g] }
    betas = rows.map { |r| r[:beta] }
    residuals = rows.map { |r| r[:residual] }
    decayed = rows.map { |r| r[:decayed_residual] }
    update = rows.map { |r| r[:update_residual] }
    taus = rows.compact_map { |r| r[:tau] }
    unstable = rows.count { |r| r[:g] >= 1.0 }
    g_rates = g_cuts.map do |cut|
      passed = rows.count { |r| r[:g] <= cut }
      "g<=#{cut.round(4)}:#{(100.0 * passed / rows.size).round(2)}%"
    end
    residual_rates = residual_thresholds.map do |threshold|
      passed = rows.count { |r| r[:residual] <= threshold }
      "r<=#{threshold.round(4)}:#{(100.0 * passed / rows.size).round(2)}%"
    end
    joint_rates = [] of String
    g_cuts.each do |cut|
      residual_thresholds.each do |threshold|
        passed = rows.count { |r| r[:g] <= cut && r[:residual] <= threshold }
        joint_rates << "g<=#{cut.round(4)}&r<=#{threshold.round(4)}:#{(100.0 * passed / rows.size).round(2)}%"
      end
    end
    tau_note =
      if taus.empty?
        "tau_finite=0 tau_p50=inf tau_p90=inf"
      else
        "tau_finite=#{taus.size} tau_p50=#{sorted_percentile(taus, 0.50).round(3)} tau_p90=#{sorted_percentile(taus, 0.90).round(3)}"
      end

    "#{prefix} samples=#{rows.size} g_mean=#{(gs.sum / gs.size).round(6)} g_p10=#{sorted_percentile(gs, 0.10).round(6)} g_p50=#{sorted_percentile(gs, 0.50).round(6)} g_p90=#{sorted_percentile(gs, 0.90).round(6)} g_min=#{gs.min.round(6)} g_max=#{gs.max.round(6)} g_ge_1=#{unstable} #{tau_note} beta_mean=#{(betas.sum / betas.size).round(6)} beta_p50=#{sorted_percentile(betas, 0.50).round(6)} residual_mean=#{(residuals.sum / residuals.size).round(6)} residual_p50=#{sorted_percentile(residuals, 0.50).round(6)} residual_p90=#{sorted_percentile(residuals, 0.90).round(6)} decayed_residual_p50=#{sorted_percentile(decayed, 0.50).round(6)} decayed_residual_p90=#{sorted_percentile(decayed, 0.90).round(6)} update_residual_p50=#{sorted_percentile(update, 0.50).round(6)} update_residual_p90=#{sorted_percentile(update, 0.90).round(6)} g_rates=#{g_rates.join(',')} residual_rates=#{residual_rates.join(',')} joint_rates=#{joint_rates.join(',')}"
  }

  notes = [] of String
  notes << format_stats.call("dn_regime_features name=#{name} layer=#{layer_index} rank=#{rank} token_vectors=#{token_count} calib_tokens=#{calib_count} heldout_tokens=#{token_count - calib_count} heads=#{h_v}", values)
  h_v.times do |h|
    head_rows = values.select { |r| r[:head] == h }
    notes << format_stats.call("dn_regime_head_features name=#{name} layer=#{layer_index} head=#{h} k_head=#{h % h_k} rank=#{rank}", head_rows)
  end
  notes
end

private def ffn_updown_route_feature_note(name : String,
                                          weights : ML::GGUF::Qwen35Weights,
                                          token_ids : Array(Int32),
                                          calib_count : Int32,
                                          layer_ids : Array(Int32),
                                          adapters : FFNUpDownAdapterMap,
                                          rank : Int32) : String
  sample_map = ffn_updown_samples_for_token_sets(weights, [token_ids], layer_ids, token_ids.size)
  rel_values = [] of Float64
  cos_values = [] of Float64
  layer_notes = [] of String

  layer_ids.uniq.sort.each do |il|
    adapter = adapters[il]? || next
    samples = sample_map[il]? || next
    eval_start = Math.min(calib_count, samples.size)
    next unless eval_start < samples.size
    layer = weights.layers[il]
    next unless layer.is_a?(ML::GGUF::Qwen35RecurrentWeights)

    layer_rel = [] of Float64
    samples[eval_start, samples.size - eval_start].each do |sample|
      activation = sample[:activation].map(&.to_f32)
      ffn_in = sample[:ffn_in].map(&.to_f32)
      exact = ML::GGUF::Qwen35CPU.qmatvec_nobias(layer.ffn_down_qw, activation)
      approx = ffn_out_from_updown_adapter(ffn_in, adapter, rank)
      err_sq = 0.0
      exact_sq = 0.0
      exact.size.times do |i|
        e = exact[i].to_f64
        d = approx[i].to_f64 - e
        err_sq += d * d
        exact_sq += e * e
      end
      rel = exact_sq > 0.0 ? Math.sqrt(err_sq / exact_sq) : 0.0
      rel_values << rel
      layer_rel << rel
      cos_values << cosine(exact, approx)
    end
    next if layer_rel.empty?
    sorted_layer = layer_rel.sort
    layer_notes << "#{il}:#{(layer_rel.sum / layer_rel.size).round(6)}/#{sorted_layer[(sorted_layer.size * 90 // 100).clamp(0, sorted_layer.size - 1)].round(6)}"
  end

  if rel_values.empty?
    return "self_spec_ffn_updown_route_features name=#{name} layers=#{layer_ids.join(',')} rank=#{rank} token_vectors=#{token_ids.size} calib_tokens=#{calib_count} heldout_samples=0"
  end

  sorted_rel = rel_values.sort
  sorted_cos = cos_values.sort
  p50 = sorted_rel[sorted_rel.size // 2]
  p90 = sorted_rel[(sorted_rel.size * 90 // 100).clamp(0, sorted_rel.size - 1)]
  p99 = sorted_rel[(sorted_rel.size * 99 // 100).clamp(0, sorted_rel.size - 1)]
  cos_mean = cos_values.sum / cos_values.size
  "self_spec_ffn_updown_route_features name=#{name} layers=#{layer_ids.join(',')} rank=#{rank} token_vectors=#{token_ids.size} calib_tokens=#{calib_count} heldout_samples=#{rel_values.size} rel_rmse_mean=#{(rel_values.sum / rel_values.size).round(6)} rel_rmse_p50=#{p50.round(6)} rel_rmse_p90=#{p90.round(6)} rel_rmse_p99=#{p99.round(6)} rel_rmse_max=#{sorted_rel[-1].round(6)} cos_mean=#{cos_mean.round(8)} cos_min=#{sorted_cos[0].round(8)} layer_rel_mean_p90=#{layer_notes.join(',')}"
end

private def project_with_basis(v : Array(Float32), offset : Int32,
                               basis : Array(Array(Float64)), rank : Int32) : Nil
  limit = Math.min(rank, basis.size)
  s = basis[0].size
  projected = Array.new(s, 0.0)
  limit.times do |i|
    b = basis[i]
    coeff = 0.0
    s.times { |d| coeff += v[offset + d].to_f64 * b[d] }
    s.times { |d| projected[d] += coeff * b[d] }
  end
  s.times { |d| v[offset + d] = projected[d].to_f32 }
end

private def residual_norm_f32(v : Array(Float32), offset : Int32,
                              basis : Array(Array(Float64)), rank : Int32) : Float64
  limit = Math.min(rank, basis.size)
  s = basis[0].size
  residual = Array.new(s) { |d| v[offset + d].to_f64 }
  limit.times do |i|
    b = basis[i]
    coeff = dot(residual, b)
    s.times { |d| residual[d] -= coeff * b[d] }
  end
  Math.sqrt(dot(residual, residual))
end

private def max_k_residual(k_conv : Array(Float32), bases : BasisSet, rank : Int32,
                           h_k : Int32, s : Int32) : Float64
  max = 0.0
  h_k.times do |h|
    residual = residual_norm_f32(k_conv, h * s, bases[h], rank)
    max = residual if residual > max
  end
  max
end

private def max_k_residual_score(k_conv : Array(Float32),
                                 ghead : Array(Float32),
                                 beta : Array(Float32),
                                 bases : BasisSet,
                                 rank : Int32,
                                 h_k : Int32,
                                 s : Int32,
                                 mode : String) : Float64
  return max_k_residual(k_conv, bases, rank, h_k, s) if mode == "raw"

  residuals = Array(Float64).new(h_k) do |h|
    residual_norm_f32(k_conv, h * s, bases[h], rank)
  end
  max = 0.0
  ghead.size.times do |h|
    weight = case mode
             when "decayed"
               ghead[h].to_f64
             when "update"
               beta[h].to_f64
             else
               raise "unknown fallback score mode #{mode.inspect}"
             end
    score = weight * residuals[h % h_k]
    max = score if score > max
  end
  max
end

private def delta_stats(exact : Array(Float32), approx : Array(Float32)) : Tuple(Float64, Float64)
  sum_sq = 0.0
  max = 0.0
  exact.size.times do |i|
    d = (exact[i] - approx[i]).to_f64.abs
    sum_sq += d * d
    max = d if d > max
  end
  {Math.sqrt(sum_sq / exact.size), max}
end

private def simulate_projected_delta(samples : Array(RecurrentSample),
                                     bases : Array(Array(Array(Float64))),
                                     rank : Int32,
                                     calib_count : Int32,
                                     h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(y_rmse: Float64, y_max: Float64, state_rmse: Float64, state_max: Float64)
  exact_state = Array(Float32).new(h_v * s * s, 0.0_f32)
  approx_state = Array(Float32).new(h_v * s * s, 0.0_f32)
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  y_exact = Array(Float32).new(h_v * s, 0.0_f32)
  y_approx = Array(Float32).new(h_v * s, 0.0_f32)
  y_sq = 0.0
  y_max = 0.0
  y_count = 0

  samples.each_with_index do |sample, idx|
    ML::GGUF::Qwen35CPU.delta_net_step!(
      exact_state, sample.q, sample.k, sample.v, sample.ghead, sample.beta,
      y_exact, h_k, h_v, s, scale
    )

    k_approx = sample.k.dup
    if idx >= calib_count
      h_k.times { |h| project_with_basis(k_approx, h * s, bases[h], rank) }
    end
    ML::GGUF::Qwen35CPU.delta_net_step!(
      approx_state, sample.q, k_approx, sample.v, sample.ghead, sample.beta,
      y_approx, h_k, h_v, s, scale
    )

    next if idx < calib_count

    rmse, max = delta_stats(y_exact, y_approx)
    y_sq += rmse * rmse * y_exact.size
    y_max = max if max > y_max
    y_count += y_exact.size
  end

  state_rmse, state_max = delta_stats(exact_state, approx_state)
  {
    y_rmse:     y_count > 0 ? Math.sqrt(y_sq / y_count) : 0.0,
    y_max:      y_max,
    state_rmse: state_rmse,
    state_max:  state_max,
  }
end

private def basis_coeffs(v : Array(Float32), offset : Int32,
                         basis : Array(Array(Float64)), rank : Int32) : Array(Float32)
  limit = Math.min(rank, basis.size)
  Array.new(limit) do |i|
    b = basis[i]
    coeff = 0.0
    b.size.times { |d| coeff += v[offset + d].to_f64 * b[d] }
    coeff.to_f32
  end
end

private def lowrank_projected_delta_step!(m_state : Array(Float32),
                                          sample : RecurrentSample,
                                          bases : Array(Array(Array(Float64))),
                                          rank : Int32,
                                          y : Array(Float32),
                                          h_k : Int32, h_v : Int32, s : Int32,
                                          scale : Float32) : Nil
  h_v.times do |h|
    k_head = h % h_k
    basis = bases[k_head]
    r = Math.min(rank, basis.size)
    q_off = k_head * s
    k_off = k_head * s
    v_off = h * s
    st_base = h * s * rank
    gh = sample.ghead[h]
    bh = sample.beta[h]
    c = basis_coeffs(sample.k, k_off, basis, r)
    qbar = basis_coeffs(sample.q, q_off, basis, r)

    s.times do |row|
      row_off = st_base + row * rank
      r.times { |j| m_state[row_off + j] *= gh }

      sk = 0.0_f32
      r.times { |j| sk += m_state[row_off + j] * c[j] }
      delt = bh * (sample.v[v_off + row] - sk)
      r.times { |j| m_state[row_off + j] += delt * c[j] }

      acc = 0.0_f32
      r.times { |j| acc += m_state[row_off + j] * qbar[j] }
      y[h * s + row] = acc * scale
    end
  end
end

private def sync_lowrank_state_from_metal!(lr_state : LowRankState) : Nil
  return unless buf = lr_state.m_buf
  return if lr_state.m.empty?
  lr_state.m = buf.read(lr_state.m.size)
end

private def flatten_basis_for_metal(bases : BasisSet, rank : Int32, h_k : Int32, s : Int32) : Array(Float32)
  flat = Array(Float32).new(h_k * rank * s, 0.0_f32)
  h_k.times do |h|
    basis = bases[h]
    r = Math.min(rank, basis.size)
    r.times do |j|
      s.times do |d|
        flat[(h * rank + j) * s + d] = basis[j][d].to_f32
      end
    end
  end
  flat
end

private def lowrank_basis_buffer!(lr_state : LowRankState, bases : BasisSet,
                                  rank : Int32, h_k : Int32, s : Int32) : ML::MetalBuffer
  key = "#{h_k}:#{s}:#{rank}:#{bases.object_id}"
  byte_size = (h_k * rank * s).to_i64 * sizeof(Float32)
  buf = lr_state.basis_buf
  if buf.nil? || buf.size != byte_size || lr_state.basis_key != key
    flat = flatten_basis_for_metal(bases, rank, h_k, s)
    buf = ML::MetalBuffer.new(byte_size)
    buf.write(flat)
    lr_state.basis_buf = buf
    lr_state.basis_key = key
  end
  buf
end

private def updown_adapter_buffers!(lr_state : LowRankState,
                                    adapter : FFNUpDownAdapter,
                                    rank : Int32,
                                    hidden_dim : Int32) : NamedTuple(x_mean: ML::MetalBuffer, c_mean: ML::MetalBuffer, coeff_w: ML::MetalBuffer, down: ML::MetalBuffer, rank: Int32)
  limit = Math.min(rank, adapter.coeff_weights.size)
  raise "FFN up/down adapter has no coefficient weights" unless limit > 0
  raise "FFN up/down adapter output dim mismatch" unless adapter.down_basis[0].size == hidden_dim
  key = "#{adapter.coeff_weights.object_id}:#{adapter.down_basis.object_id}:#{limit}:#{hidden_dim}"
  byte_size = (limit * hidden_dim).to_i64 * sizeof(Float32)
  needs_upload = lr_state.updown_key != key ||
                 lr_state.updown_x_mean_buf.nil? ||
                 lr_state.updown_c_mean_buf.nil? ||
                 lr_state.updown_coeff_w_buf.nil? ||
                 lr_state.updown_down_buf.nil? ||
                 lr_state.updown_coeff_w_buf.not_nil!.size != byte_size ||
                 lr_state.updown_down_buf.not_nil!.size != byte_size
  if needs_upload
    coeff_weights = Array(Float32).new(limit * hidden_dim)
    down_basis = Array(Float32).new(limit * hidden_dim)
    limit.times do |j|
      hidden_dim.times { |d| coeff_weights << adapter.coeff_weights[j][d].to_f32 }
      hidden_dim.times { |d| down_basis << adapter.down_basis[j][d] }
    end
    lr_state.updown_x_mean_buf = ML::MetalBuffer.from_array(adapter.x_mean.map(&.to_f32))
    lr_state.updown_c_mean_buf = ML::MetalBuffer.from_array(adapter.c_mean[0, limit].map(&.to_f32))
    lr_state.updown_coeff_w_buf = ML::MetalBuffer.from_array(coeff_weights)
    lr_state.updown_down_buf = ML::MetalBuffer.from_array(down_basis)
    lr_state.updown_key = key
  end
  {
    x_mean:  lr_state.updown_x_mean_buf.not_nil!,
    c_mean:  lr_state.updown_c_mean_buf.not_nil!,
    coeff_w: lr_state.updown_coeff_w_buf.not_nil!,
    down:    lr_state.updown_down_buf.not_nil!,
    rank:    limit,
  }
end

private def build_updown_adapter_buffer_maps(adapters : FFNUpDownAdapterMap,
                                             layer_ids : Enumerable(Int32),
                                             rank : Int32,
                                             hidden_dim : Int32) : NamedTuple(x_mean: Hash(Int32, ML::MetalBuffer), c_mean: Hash(Int32, ML::MetalBuffer), coeff_w: Hash(Int32, ML::MetalBuffer), down: Hash(Int32, ML::MetalBuffer), rank: Int32)
  ML::GGUF::Qwen35SelfSpecUpdownBuffers.build_f32_maps(adapters, layer_ids, rank, hidden_dim)
end

private def quantize_row_q8(values : Array(Float32)) : NamedTuple(bytes: Bytes, scale: Float32)
  max_abs = 0.0_f32
  values.each do |v|
    a = v.abs
    max_abs = a if a > max_abs
  end
  scale = max_abs > 0.0_f32 ? (max_abs / 127.0_f32) : 1.0_f32
  bytes = Bytes.new(values.size)
  values.each_with_index do |v, i|
    q = (v / scale).round.to_i
    q = -127 if q < -127
    q = 127 if q > 127
    bytes[i] = (q < 0 ? q + 256 : q).to_u8
  end
  {bytes: bytes, scale: scale}
end

private def build_updown_adapter_q8_buffer_maps(adapters : FFNUpDownAdapterMap,
                                                layer_ids : Enumerable(Int32),
                                                rank : Int32,
                                                hidden_dim : Int32) : NamedTuple(x_mean: Hash(Int32, ML::MetalBuffer), c_mean: Hash(Int32, ML::MetalBuffer), coeff_q8: Hash(Int32, ML::MetalBuffer), coeff_scales: Hash(Int32, ML::MetalBuffer), down_q8: Hash(Int32, ML::MetalBuffer), down_scales: Hash(Int32, ML::MetalBuffer), rank: Int32)
  raise "GPU pipeline pca-updown q8 rank must be positive" unless rank > 0
  raise "GPU pipeline pca-updown q8 rank too large for current Metal kernel" if rank > 64

  x_mean = {} of Int32 => ML::MetalBuffer
  c_mean = {} of Int32 => ML::MetalBuffer
  coeff_q8 = {} of Int32 => ML::MetalBuffer
  coeff_scales = {} of Int32 => ML::MetalBuffer
  down_q8 = {} of Int32 => ML::MetalBuffer
  down_scales = {} of Int32 => ML::MetalBuffer
  actual_rank = nil.as(Int32?)
  layer_ids.each do |il|
    adapter = adapters[il]? || raise "GPU pipeline pca-updown q8 missing adapter for layer #{il}"
    limit = Math.min(rank, adapter.coeff_weights.size)
    raise "GPU pipeline pca-updown q8 has no coefficient weights" unless limit > 0
    raise "GPU pipeline pca-updown q8 output dim mismatch" unless adapter.down_basis[0].size == hidden_dim
    if prev_rank = actual_rank
      raise "GPU pipeline pca-updown q8 inconsistent adapter ranks: #{prev_rank} vs #{limit} at layer #{il}" unless prev_rank == limit
    else
      actual_rank = limit
    end

    coeff_bytes = Bytes.new(limit * hidden_dim)
    down_bytes = Bytes.new(limit * hidden_dim)
    coeff_scale_values = Array(Float32).new(limit)
    down_scale_values = Array(Float32).new(limit)
    limit.times do |j|
      coeff_row = Array(Float32).new(hidden_dim)
      hidden_dim.times { |d| coeff_row << adapter.coeff_weights[j][d].to_f32 }
      cq = quantize_row_q8(coeff_row)
      coeff_scale_values << cq[:scale]
      cq[:bytes].each_with_index { |b, d| coeff_bytes[j * hidden_dim + d] = b }

      down_row = Array(Float32).new(hidden_dim)
      hidden_dim.times { |d| down_row << adapter.down_basis[j][d] }
      dq = quantize_row_q8(down_row)
      down_scale_values << dq[:scale]
      dq[:bytes].each_with_index { |b, d| down_bytes[j * hidden_dim + d] = b }
    end

    coeff_buf = ML::MetalBuffer.new(coeff_bytes.size.to_i64)
    coeff_buf.write_bytes(coeff_bytes.to_unsafe, coeff_bytes.size)
    down_buf = ML::MetalBuffer.new(down_bytes.size.to_i64)
    down_buf.write_bytes(down_bytes.to_unsafe, down_bytes.size)

    x_mean[il] = ML::MetalBuffer.from_array(adapter.x_mean.map(&.to_f32))
    c_mean[il] = ML::MetalBuffer.from_array(adapter.c_mean[0, limit].map(&.to_f32))
    coeff_q8[il] = coeff_buf
    coeff_scales[il] = ML::MetalBuffer.from_array(coeff_scale_values)
    down_q8[il] = down_buf
    down_scales[il] = ML::MetalBuffer.from_array(down_scale_values)
  end
  {
    x_mean:       x_mean,
    c_mean:       c_mean,
    coeff_q8:     coeff_q8,
    coeff_scales: coeff_scales,
    down_q8:      down_q8,
    down_scales:  down_scales,
    rank:         actual_rank || raise("GPU pipeline pca-updown q8 has no layers"),
  }
end

private def lowrank_state_buffer!(lr_state : LowRankState) : ML::MetalBuffer
  byte_size = lr_state.m.size.to_i64 * sizeof(Float32)
  buf = lr_state.m_buf
  if buf.nil? || buf.size != byte_size
    buf = ML::MetalBuffer.new(byte_size)
    lr_state.m_buf = buf
    lr_state.full_state_current = true
  end
  buf.write(lr_state.m) if lr_state.full_state_current
  buf
end

private def lowrank_projected_delta_step_metal!(lr_state : LowRankState,
                                                sample : RecurrentSample,
                                                bases : Array(Array(Array(Float64))),
                                                rank : Int32,
                                                y : Array(Float32),
                                                h_k : Int32, h_v : Int32, s : Int32,
                                                scale : Float32,
                                                project_coeffs_on_gpu : Bool = false) : Nil
  raise "Metal low-rank delta unavailable" unless ML::GGUF::Qwen35Metal.available?

  buf = lowrank_state_buffer!(lr_state)

  y_metal = if project_coeffs_on_gpu
              basis_buf = lowrank_basis_buffer!(lr_state, bases, rank, h_k, s)
              ML::GGUF::Qwen35Metal.lowrank_delta_step_projected_buf(buf, sample.q, sample.k, basis_buf, sample.v, sample.ghead, sample.beta,
                h_k, h_v, s, rank, scale)
            else
              c = Array(Float32).new(h_k * rank, 0.0_f32)
              qbar = Array(Float32).new(h_k * rank, 0.0_f32)
              h_k.times do |h|
                basis = bases[h]
                r = Math.min(rank, basis.size)
                k_coeffs = basis_coeffs(sample.k, h * s, basis, r)
                q_coeffs = basis_coeffs(sample.q, h * s, basis, r)
                r.times do |j|
                  c[h * rank + j] = k_coeffs[j]
                  qbar[h * rank + j] = q_coeffs[j]
                end
              end
              ML::GGUF::Qwen35Metal.lowrank_delta_step(buf, c, qbar, sample.v, sample.ghead, sample.beta,
                h_k, h_v, s, rank, scale)
            end
  y_metal.each_with_index { |v, i| y[i] = v }
end

private def reconstruct_lowrank_state(m_state : Array(Float32),
                                      bases : Array(Array(Array(Float64))),
                                      rank : Int32,
                                      h_k : Int32, h_v : Int32, s : Int32) : Array(Float32)
  out = Array(Float32).new(h_v * s * s, 0.0_f32)
  h_v.times do |h|
    basis = bases[h % h_k]
    r = Math.min(rank, basis.size)
    m_base = h * s * rank
    out_base = h * s * s
    s.times do |row|
      r.times do |j|
        coeff = m_state[m_base + row * rank + j].to_f64
        b = basis[j]
        s.times { |d| out[out_base + row * s + d] += (coeff * b[d]).to_f32 }
      end
    end
  end
  out
end

private def simulate_lowrank_projected_delta(samples : Array(RecurrentSample),
                                             bases : Array(Array(Array(Float64))),
                                             rank : Int32,
                                             calib_count : Int32,
                                             h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(exact_y_rmse: Float64, exact_y_max: Float64, proof_y_rmse: Float64, proof_y_max: Float64, proof_state_rmse: Float64, proof_state_max: Float64)
  exact_state = Array(Float32).new(h_v * s * s, 0.0_f32)
  projected_state = Array(Float32).new(h_v * s * s, 0.0_f32)
  lowrank_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  y_exact = Array(Float32).new(h_v * s, 0.0_f32)
  y_projected = Array(Float32).new(h_v * s, 0.0_f32)
  y_lowrank = Array(Float32).new(h_v * s, 0.0_f32)
  exact_sq = 0.0
  proof_sq = 0.0
  exact_max = 0.0
  proof_max = 0.0
  count = 0

  samples.each_with_index do |sample, idx|
    ML::GGUF::Qwen35CPU.delta_net_step!(
      exact_state, sample.q, sample.k, sample.v, sample.ghead, sample.beta,
      y_exact, h_k, h_v, s, scale
    )

    k_projected = sample.k.dup
    h_k.times { |h| project_with_basis(k_projected, h * s, bases[h], rank) }
    ML::GGUF::Qwen35CPU.delta_net_step!(
      projected_state, sample.q, k_projected, sample.v, sample.ghead, sample.beta,
      y_projected, h_k, h_v, s, scale
    )

    lowrank_projected_delta_step!(
      lowrank_state, sample, bases, rank, y_lowrank, h_k, h_v, s, scale
    )

    next if idx < calib_count

    exact_rmse, exact_step_max = delta_stats(y_exact, y_projected)
    proof_rmse, proof_step_max = delta_stats(y_projected, y_lowrank)
    exact_sq += exact_rmse * exact_rmse * y_exact.size
    proof_sq += proof_rmse * proof_rmse * y_exact.size
    exact_max = exact_step_max if exact_step_max > exact_max
    proof_max = proof_step_max if proof_step_max > proof_max
    count += y_exact.size
  end

  reconstructed = reconstruct_lowrank_state(lowrank_state, bases, rank, h_k, h_v, s)
  proof_state_rmse, proof_state_max = delta_stats(projected_state, reconstructed)
  {
    exact_y_rmse:     count > 0 ? Math.sqrt(exact_sq / count) : 0.0,
    exact_y_max:      exact_max,
    proof_y_rmse:     count > 0 ? Math.sqrt(proof_sq / count) : 0.0,
    proof_y_max:      proof_max,
    proof_state_rmse: proof_state_rmse,
    proof_state_max:  proof_state_max,
  }
end

private def simulate_lowrank_projected_delta_metal(samples : Array(RecurrentSample),
                                                   bases : Array(Array(Array(Float64))),
                                                   rank : Int32,
                                                   calib_count : Int32,
                                                   h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(y_rmse: Float64, y_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank delta unavailable" unless ML::GGUF::Qwen35Metal.available?
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)

  y_sq = 0.0
  y_max = 0.0
  count = 0
  steps = 0
  cpu_ms = 0.0
  metal_ms = 0.0
  samples[calib_count, samples.size - calib_count].each do |sample|
    c = Array(Float32).new(h_k * rank, 0.0_f32)
    qbar = Array(Float32).new(h_k * rank, 0.0_f32)
    h_k.times do |h|
      basis = bases[h]
      r = Math.min(rank, basis.size)
      k_coeffs = basis_coeffs(sample.k, h * s, basis, r)
      q_coeffs = basis_coeffs(sample.q, h * s, basis, r)
      r.times do |j|
        c[h * rank + j] = k_coeffs[j]
        qbar[h * rank + j] = q_coeffs[j]
      end
    end

    t_cpu = Time.instant
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    cpu_ms += (Time.instant - t_cpu).total_milliseconds
    t_metal = Time.instant
    y_metal = ML::GGUF::Qwen35Metal.lowrank_delta_step(metal_buf, c, qbar, sample.v, sample.ghead, sample.beta, h_k, h_v, s, rank, scale)
    metal_ms += (Time.instant - t_metal).total_milliseconds
    y_cpu.each_with_index do |v, i|
      e = (v - y_metal[i]).abs.to_f64
      y_sq += e * e
      y_max = e if e > y_max
      count += 1
    end
    steps += 1
  end

  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  {
    y_rmse:     count > 0 ? Math.sqrt(y_sq / count) : 0.0,
    y_max:      y_max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      steps,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def simulate_lowrank_projected_delta_metal_project(samples : Array(RecurrentSample),
                                                           bases : BasisSet,
                                                           rank : Int32,
                                                           calib_count : Int32,
                                                           h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(y_rmse: Float64, y_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank projected delta unavailable" unless ML::GGUF::Qwen35Metal.available?
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)

  y_sq = 0.0
  y_max = 0.0
  count = 0
  steps = 0
  cpu_ms = 0.0
  metal_ms = 0.0
  samples[calib_count, samples.size - calib_count].each do |sample|
    t_cpu = Time.instant
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    cpu_ms += (Time.instant - t_cpu).total_milliseconds

    t_metal = Time.instant
    y_metal = ML::GGUF::Qwen35Metal.lowrank_delta_step_projected_buf(metal_buf, sample.q, sample.k, basis_buf, sample.v, sample.ghead, sample.beta,
      h_k, h_v, s, rank, scale)
    metal_ms += (Time.instant - t_metal).total_milliseconds

    y_cpu.each_with_index do |v, i|
      e = (v - y_metal[i]).abs.to_f64
      y_sq += e * e
      y_max = e if e > y_max
      count += 1
    end
    steps += 1
  end

  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  {
    y_rmse:     count > 0 ? Math.sqrt(y_sq / count) : 0.0,
    y_max:      y_max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      steps,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def simulate_lowrank_projected_delta_metal_chunk(samples : Array(RecurrentSample),
                                                         bases : BasisSet,
                                                         rank : Int32,
                                                         calib_count : Int32,
                                                         h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(y_rmse: Float64, y_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank delta unavailable" unless ML::GGUF::Qwen35Metal.available?
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)

  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)

  heldout.each_with_index do |sample, t|
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times { |i| v_all[t * h_v * s + i] = sample.v[i] }
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end

  t_cpu = Time.instant
  heldout.each_with_index do |sample, t|
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    y_cpu.each_with_index { |v, i| y_cpu_all[t * h_v * s + i] = v }
  end
  cpu_ms = (Time.instant - t_cpu).total_milliseconds

  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  t_metal = Time.instant
  y_metal_all = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_buf(metal_buf, q_all, k_all, basis_buf, v_all, g_all, b_all,
    h_k, h_v, s, rank, n_tokens, scale)
  metal_ms = (Time.instant - t_metal).total_milliseconds

  y_sq = 0.0
  y_max = 0.0
  y_cpu_all.each_with_index do |v, i|
    e = (v - y_metal_all[i]).abs.to_f64
    y_sq += e * e
    y_max = e if e > y_max
  end
  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  count = y_cpu_all.size
  {
    y_rmse:     count > 0 ? Math.sqrt(y_sq / count) : 0.0,
    y_max:      y_max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      n_tokens,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def simulate_lowrank_projected_delta_metal_chunk_out(samples : Array(RecurrentSample),
                                                             bases : BasisSet,
                                                             out_qw : ML::GGUF::QuantWeight,
                                                             ssm_norm : Array(Float32),
                                                             eps : Float32,
                                                             rank : Int32,
                                                             calib_count : Int32,
                                                             h_k : Int32, h_v : Int32, s : Int32) : NamedTuple(out_rmse: Float64, out_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank delta unavailable" unless ML::GGUF::Qwen35Metal.available?
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)
  out_cpu_all = Array(Float32).new(n_tokens * out_qw.out_dim, 0.0_f32)

  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  z_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)

  heldout.each_with_index do |sample, t|
    raise "sample z missing for chunk_out proof" if sample.z.empty?
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times do |i|
      v_all[t * h_v * s + i] = sample.v[i]
      z_all[t * h_v * s + i] = sample.z[i]
    end
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end

  t_cpu = Time.instant
  heldout.each_with_index do |sample, t|
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y_cpu, h * s, s, ssm_norm, eps) }
    (h_v * s).times { |i| y_cpu[i] = y_cpu[i] * ML::GGUF::Qwen35CPU.silu(sample.z[i]) }
    out = ML::GGUF::Qwen35CPU.qmatvec_nobias(out_qw, y_cpu)
    out.each_with_index { |v, i| out_cpu_all[t * out_qw.out_dim + i] = v }
  end
  cpu_ms = (Time.instant - t_cpu).total_milliseconds

  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  t_metal = Time.instant
  out_metal_all = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_out_buf(metal_buf, q_all, k_all, basis_buf, v_all, g_all, b_all, z_all,
    ssm_norm, out_qw, h_k, h_v, s, rank, n_tokens, eps, scale).not_nil!
  metal_ms = (Time.instant - t_metal).total_milliseconds

  sq = 0.0
  max = 0.0
  out_cpu_all.each_with_index do |v, i|
    e = (v - out_metal_all[i]).abs.to_f64
    sq += e * e
    max = e if e > max
  end
  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  count = out_cpu_all.size
  {
    out_rmse:   count > 0 ? Math.sqrt(sq / count) : 0.0,
    out_max:    max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      n_tokens,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def finish_recurrent_layer_cpu(inp : Array(Float32),
                                       attn_out : Array(Float32),
                                       lw : ML::GGUF::Qwen35RecurrentWeights,
                                       hp : ML::GGUF::Qwen35Hparams) : Array(Float32)
  inp_l2 = Array(Float32).new(hp.n_embd) { |i| inp[i] + attn_out[i] }
  ffn_in = ML::GGUF::Qwen35CPU.rms_norm(inp_l2, lw.post_attention_norm, hp.rms_eps)
  gate_up = ML::GGUF::Qwen35CPU.qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in)
  gate = gate_up[0]
  up = gate_up[1]
  combined = Array(Float32).new(hp.n_ff) { |i| ML::GGUF::Qwen35CPU.silu(gate[i]) * up[i] }
  ffn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ffn_down_qw, combined)
  Array(Float32).new(hp.n_embd) { |i| inp_l2[i] + ffn_out[i] }
end

private def simulate_lowrank_recurrent_layer_metal_chunk(samples : Array(RecurrentSample),
                                                         bases : BasisSet,
                                                         lw : ML::GGUF::Qwen35RecurrentWeights,
                                                         hp : ML::GGUF::Qwen35Hparams,
                                                         rank : Int32,
                                                         calib_count : Int32) : NamedTuple(layer_rmse: Float64, layer_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank recurrent layer chunk unavailable" unless ML::GGUF::Qwen35Metal.available?
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)
  layer_cpu_all = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)

  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  z_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)

  heldout.each_with_index do |sample, t|
    raise "sample inp missing for layer chunk proof" if sample.inp.empty?
    raise "sample z missing for layer chunk proof" if sample.z.empty?
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times do |i|
      v_all[t * h_v * s + i] = sample.v[i]
      z_all[t * h_v * s + i] = sample.z[i]
    end
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end

  t_cpu = Time.instant
  heldout.each_with_index do |sample, t|
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y_cpu, h * s, s, lw.ssm_norm, hp.rms_eps) }
    (h_v * s).times { |i| y_cpu[i] = y_cpu[i] * ML::GGUF::Qwen35CPU.silu(sample.z[i]) }
    attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y_cpu)
    out = finish_recurrent_layer_cpu(sample.inp, attn_out, lw, hp)
    out.each_with_index { |v, i| layer_cpu_all[t * hp.n_embd + i] = v }
  end
  cpu_ms = (Time.instant - t_cpu).total_milliseconds

  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  t_metal = Time.instant
  attn_metal_all = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_out_buf(metal_buf, q_all, k_all, basis_buf, v_all, g_all, b_all, z_all,
    lw.ssm_norm, lw.ssm_out_qw, h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  layer_metal_all = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)
  heldout.each_with_index do |sample, t|
    attn = attn_metal_all[t * hp.n_embd, hp.n_embd]
    out = finish_recurrent_layer_cpu(sample.inp, attn, lw, hp)
    out.each_with_index { |v, i| layer_metal_all[t * hp.n_embd + i] = v }
  end
  metal_ms = (Time.instant - t_metal).total_milliseconds

  sq = 0.0
  max = 0.0
  layer_cpu_all.each_with_index do |v, i|
    e = (v - layer_metal_all[i]).abs.to_f64
    sq += e * e
    max = e if e > max
  end
  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  count = layer_cpu_all.size
  {
    layer_rmse: count > 0 ? Math.sqrt(sq / count) : 0.0,
    layer_max:  max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      n_tokens,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def simulate_lowrank_recurrent_layer_full_metal_chunk(samples : Array(RecurrentSample),
                                                              bases : BasisSet,
                                                              lw : ML::GGUF::Qwen35RecurrentWeights,
                                                              hp : ML::GGUF::Qwen35Hparams,
                                                              rank : Int32,
                                                              calib_count : Int32) : NamedTuple(layer_rmse: Float64, layer_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64)
  raise "Metal low-rank recurrent full layer chunk unavailable" unless ML::GGUF::Qwen35Metal.available?
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)
  layer_cpu_all = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)

  inp_all = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)
  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  z_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)

  heldout.each_with_index do |sample, t|
    raise "sample inp missing for full layer chunk proof" if sample.inp.empty?
    raise "sample z missing for full layer chunk proof" if sample.z.empty?
    hp.n_embd.times { |i| inp_all[t * hp.n_embd + i] = sample.inp[i] }
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times do |i|
      v_all[t * h_v * s + i] = sample.v[i]
      z_all[t * h_v * s + i] = sample.z[i]
    end
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end

  t_cpu = Time.instant
  heldout.each_with_index do |sample, t|
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y_cpu, h * s, s, lw.ssm_norm, hp.rms_eps) }
    (h_v * s).times { |i| y_cpu[i] = y_cpu[i] * ML::GGUF::Qwen35CPU.silu(sample.z[i]) }
    attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y_cpu)
    out = finish_recurrent_layer_cpu(sample.inp, attn_out, lw, hp)
    out.each_with_index { |v, i| layer_cpu_all[t * hp.n_embd + i] = v }
  end
  cpu_ms = (Time.instant - t_cpu).total_milliseconds

  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  t_metal = Time.instant
  layer_metal_all = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(metal_buf, inp_all, q_all, k_all, basis_buf, v_all, g_all, b_all, z_all,
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  metal_ms = (Time.instant - t_metal).total_milliseconds

  sq = 0.0
  max = 0.0
  layer_cpu_all.each_with_index do |v, i|
    e = (v - layer_metal_all[i]).abs.to_f64
    sq += e * e
    max = e if e > max
  end
  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  count = layer_cpu_all.size
  {
    layer_rmse: count > 0 ? Math.sqrt(sq / count) : 0.0,
    layer_max:  max,
    state_rmse: state_rmse,
    state_max:  state_max,
    steps:      n_tokens,
    cpu_ms:     cpu_ms,
    metal_ms:   metal_ms,
  }
end

private def simulate_lowrank_recurrent_layer_updown_metal_chunk(samples : Array(RecurrentSample),
                                                                bases : BasisSet,
                                                                lw : ML::GGUF::Qwen35RecurrentWeights,
                                                                hp : ML::GGUF::Qwen35Hparams,
                                                                rank : Int32,
                                                                calib_count : Int32,
                                                                adapter : FFNUpDownAdapter,
                                                                updown_rank : Int32) : NamedTuple(layer_rmse: Float64, layer_max: Float64, state_rmse: Float64, state_max: Float64, steps: Int32, cpu_ms: Float64, metal_ms: Float64, updown_rank: Int32)
  raise "Metal low-rank recurrent updown layer chunk unavailable" unless ML::GGUF::Qwen35Metal.available?
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  hidden_dim = hp.n_embd
  bench_rank = Math.min(updown_rank, adapter.coeff_weights.size)
  raise "updown layer rank must be positive" unless bench_rank > 0
  raise "updown layer hidden mismatch" unless adapter.down_basis[0].size == hidden_dim

  cpu_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_state = Array(Float32).new(h_v * s * rank, 0.0_f32)
  metal_buf = ML::MetalBuffer.new(metal_state.size.to_i64 * sizeof(Float32))
  metal_buf.write(metal_state)
  y_cpu = Array(Float32).new(h_v * s, 0.0_f32)
  layer_cpu_all = Array(Float32).new(n_tokens * hidden_dim, 0.0_f32)

  inp_all = Array(Float32).new(n_tokens * hidden_dim, 0.0_f32)
  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  z_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)

  heldout.each_with_index do |sample, t|
    raise "sample inp missing for updown layer chunk proof" if sample.inp.empty?
    raise "sample z missing for updown layer chunk proof" if sample.z.empty?
    hidden_dim.times { |i| inp_all[t * hidden_dim + i] = sample.inp[i] }
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times do |i|
      v_all[t * h_v * s + i] = sample.v[i]
      z_all[t * h_v * s + i] = sample.z[i]
    end
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end

  t_cpu = Time.instant
  heldout.each_with_index do |sample, t|
    lowrank_projected_delta_step!(cpu_state, sample, bases, rank, y_cpu, h_k, h_v, s, scale)
    h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y_cpu, h * s, s, lw.ssm_norm, hp.rms_eps) }
    (h_v * s).times { |i| y_cpu[i] = y_cpu[i] * ML::GGUF::Qwen35CPU.silu(sample.z[i]) }
    attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y_cpu)
    inp_l2 = Array(Float32).new(hidden_dim) { |i| sample.inp[i] + attn_out[i] }
    ffn_in = ML::GGUF::Qwen35CPU.rms_norm(inp_l2, lw.post_attention_norm, hp.rms_eps)
    ffn_out = ffn_out_from_updown_adapter(ffn_in, adapter, bench_rank)
    hidden_dim.times { |i| layer_cpu_all[t * hidden_dim + i] = inp_l2[i] + ffn_out[i] }
  end
  cpu_ms = (Time.instant - t_cpu).total_milliseconds

  coeff_weights = Array(Float32).new(bench_rank * hidden_dim)
  down_basis = Array(Float32).new(bench_rank * hidden_dim)
  bench_rank.times do |j|
    hidden_dim.times { |d| coeff_weights << adapter.coeff_weights[j][d].to_f32 }
    hidden_dim.times { |d| down_basis << adapter.down_basis[j][d] }
  end
  x_mean_buf = ML::MetalBuffer.from_array(adapter.x_mean.map(&.to_f32))
  c_mean_buf = ML::MetalBuffer.from_array(adapter.c_mean[0, bench_rank].map(&.to_f32))
  coeff_w_buf = ML::MetalBuffer.from_array(coeff_weights)
  down_buf = ML::MetalBuffer.from_array(down_basis)

  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  t_metal = Time.instant
  layer_metal_all = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_updown_buf(metal_buf, inp_all, q_all, k_all, basis_buf, v_all, g_all, b_all, z_all,
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    x_mean_buf, c_mean_buf, coeff_w_buf, down_buf,
    h_k, h_v, s, rank, n_tokens, bench_rank, hp.rms_eps.to_f32, scale).not_nil!
  metal_ms = (Time.instant - t_metal).total_milliseconds

  sq = 0.0
  max = 0.0
  layer_cpu_all.each_with_index do |v, i|
    e = (v - layer_metal_all[i]).abs.to_f64
    sq += e * e
    max = e if e > max
  end
  metal_state = metal_buf.read(metal_state.size)
  state_rmse, state_max = delta_stats(cpu_state, metal_state)
  count = layer_cpu_all.size
  {
    layer_rmse:  count > 0 ? Math.sqrt(sq / count) : 0.0,
    layer_max:   max,
    state_rmse:  state_rmse,
    state_max:   state_max,
    steps:       n_tokens,
    cpu_ms:      cpu_ms,
    metal_ms:    metal_ms,
    updown_rank: bench_rank,
  }
end

private def lowrank_layer_chunk_inputs(samples : Array(RecurrentSample),
                                       calib_count : Int32,
                                       h_k : Int32, h_v : Int32, s : Int32,
                                       hidden_dim : Int32) : NamedTuple(n_tokens: Int32, inp: Array(Float32), q: Array(Float32), k: Array(Float32), v: Array(Float32), g: Array(Float32), beta: Array(Float32), z: Array(Float32))
  heldout = samples[calib_count, samples.size - calib_count]
  n_tokens = heldout.size
  inp_all = Array(Float32).new(n_tokens * hidden_dim, 0.0_f32)
  q_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  k_all = Array(Float32).new(n_tokens * h_k * s, 0.0_f32)
  v_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  g_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  b_all = Array(Float32).new(n_tokens * h_v, 0.0_f32)
  z_all = Array(Float32).new(n_tokens * h_v * s, 0.0_f32)
  heldout.each_with_index do |sample, t|
    raise "sample inp missing for layer chunk inputs" if sample.inp.empty?
    raise "sample z missing for layer chunk inputs" if sample.z.empty?
    hidden_dim.times { |i| inp_all[t * hidden_dim + i] = sample.inp[i] }
    (h_k * s).times do |i|
      q_all[t * h_k * s + i] = sample.q[i]
      k_all[t * h_k * s + i] = sample.k[i]
    end
    (h_v * s).times do |i|
      v_all[t * h_v * s + i] = sample.v[i]
      z_all[t * h_v * s + i] = sample.z[i]
    end
    h_v.times do |i|
      g_all[t * h_v + i] = sample.ghead[i]
      b_all[t * h_v + i] = sample.beta[i]
    end
  end
  {n_tokens: n_tokens, inp: inp_all, q: q_all, k: k_all, v: v_all, g: g_all, beta: b_all, z: z_all}
end

private def simulate_lowrank_recurrent_layer_full_async_overlap(samples : Array(RecurrentSample),
                                                                bases : BasisSet,
                                                                lw : ML::GGUF::Qwen35RecurrentWeights,
                                                                hp : ML::GGUF::Qwen35Hparams,
                                                                rank : Int32,
                                                                calib_count : Int32) : NamedTuple(steps: Int32, serial_ms: Float64, async_ms: Float64, speedup: Float64, output_max: Float64)
  raise "Metal low-rank recurrent async overlap unavailable" unless ML::GGUF::Qwen35Metal.available?
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  inputs = lowrank_layer_chunk_inputs(samples, calib_count, h_k, h_v, s, hp.n_embd)
  n_tokens = inputs[:n_tokens]
  state_size = h_v * s * rank
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))

  state_serial_a = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  state_serial_b = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  state_serial_a.write(Array(Float32).new(state_size, 0.0_f32))
  state_serial_b.write(Array(Float32).new(state_size, 0.0_f32))
  t_serial = Time.instant
  serial_a = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(state_serial_a, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(state_serial_b, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  serial_ms = (Time.instant - t_serial).total_milliseconds

  state_async_a = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  state_async_b = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  state_async_a.write(Array(Float32).new(state_size, 0.0_f32))
  state_async_b.write(Array(Float32).new(state_size, 0.0_f32))
  t_async = Time.instant
  sub_a = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(state_async_a, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "lr_async_a", command_queue_name: "lr_async_a").not_nil!
  sub_b = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(state_async_b, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "lr_async_b", command_queue_name: "lr_async_b").not_nil!
  async_a = ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(sub_a)
  ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(sub_b)
  async_ms = (Time.instant - t_async).total_milliseconds

  max = 0.0
  serial_a.each_with_index do |v, i|
    e = (v - async_a[i]).abs.to_f64
    max = e if e > max
  end
  {steps: n_tokens, serial_ms: serial_ms, async_ms: async_ms, speedup: async_ms > 0.0 ? serial_ms / async_ms : 0.0, output_max: max}
end

private def verifier_state_after_prefix(weights : ML::GGUF::Qwen35Weights,
                                        prefix_ids : Array(Int32),
                                        max_seq : Int32) : ML::GGUF::Qwen35CPU::State
  hp = weights.hparams
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
  ML::GGUF::Qwen35CPU.prefill_tokens(weights, prefix_ids, 0, state) unless prefix_ids.empty?
  state
end

private def simulate_lowrank_draft_exact_verifier_overlap(samples : Array(RecurrentSample),
                                                          bases : BasisSet,
                                                          weights : ML::GGUF::Qwen35Weights,
                                                          token_ids : Array(Int32),
                                                          lw : ML::GGUF::Qwen35RecurrentWeights,
                                                          hp : ML::GGUF::Qwen35Hparams,
                                                          rank : Int32,
                                                          calib_count : Int32) : NamedTuple(steps: Int32, draft_ms: Float64, verifier_ms: Float64, serial_ms: Float64, overlap_ms: Float64, speedup: Float64, hidden_ms: Float64, draft_output_max: Float64, verifier_match: Bool)
  raise "Metal low-rank draft/verifier overlap unavailable" unless ML::GGUF::Qwen35Metal.available?
  raise "calib_count must leave a non-empty verifier span" unless calib_count < token_ids.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  inputs = lowrank_layer_chunk_inputs(samples, calib_count, h_k, h_v, s, hp.n_embd)
  n_tokens = inputs[:n_tokens]
  candidates = token_ids[calib_count, n_tokens]
  prefix_ids = token_ids[0, calib_count]
  max_seq = token_ids.size + n_tokens + 8
  state_size = h_v * s * rank
  zero_state = Array(Float32).new(state_size, 0.0_f32)
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))

  # Warm both routes outside the measured region so the comparison focuses on
  # scheduling overlap rather than one-time pipeline/constant cache setup.
  warm_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, warm_state)
  warm_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  warm_draft_state.write(zero_state)
  ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(warm_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!

  serial_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  serial_draft_state.write(zero_state)
  serial_verifier_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_draft = Time.instant
  serial_draft = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(serial_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  draft_ms = (Time.instant - t_draft).total_milliseconds
  t_verify = Time.instant
  serial_verifier = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, serial_verifier_state)
  verifier_ms = (Time.instant - t_verify).total_milliseconds
  serial_ms = draft_ms + verifier_ms

  overlap_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  overlap_draft_state.write(zero_state)
  overlap_verifier_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_overlap = Time.instant
  sub = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(overlap_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "lr_verifier_draft", command_queue_name: "lr_verifier_draft").not_nil!
  overlap_verifier = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, overlap_verifier_state)
  overlap_draft = ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(sub)
  overlap_ms = (Time.instant - t_overlap).total_milliseconds

  max = 0.0
  serial_draft.each_with_index do |v, i|
    e = (v - overlap_draft[i]).abs.to_f64
    max = e if e > max
  end
  {
    steps:            n_tokens,
    draft_ms:         draft_ms,
    verifier_ms:      verifier_ms,
    serial_ms:        serial_ms,
    overlap_ms:       overlap_ms,
    speedup:          overlap_ms > 0.0 ? serial_ms / overlap_ms : 0.0,
    hidden_ms:        serial_ms - overlap_ms,
    draft_output_max: max,
    verifier_match:   serial_verifier == overlap_verifier,
  }
end

private def simulate_lowrank_draft_exact_decode_verifier_overlap(samples : Array(RecurrentSample),
                                                                 bases : BasisSet,
                                                                 weights : ML::GGUF::Qwen35Weights,
                                                                 token_ids : Array(Int32),
                                                                 lw : ML::GGUF::Qwen35RecurrentWeights,
                                                                 hp : ML::GGUF::Qwen35Hparams,
                                                                 rank : Int32,
                                                                 calib_count : Int32) : NamedTuple(steps: Int32, draft_ms: Float64, verifier_serial_ms: Float64, verifier_async_ms: Float64, overlap_ms: Float64, async_speedup: Float64, overlap_speedup: Float64, hidden_ms: Float64, draft_output_max: Float64, verifier_match: Bool)
  raise "Metal exact decode verifier overlap unavailable" unless ML::GGUF::Qwen35Metal.available?
  raise "calib_count must leave a non-empty verifier span" unless calib_count < token_ids.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  inputs = lowrank_layer_chunk_inputs(samples, calib_count, h_k, h_v, s, hp.n_embd)
  n_tokens = inputs[:n_tokens]
  candidates = token_ids[calib_count, n_tokens]
  prefix_ids = token_ids[0, calib_count]
  max_seq = token_ids.size + n_tokens + 8
  state_size = h_v * s * rank
  zero_state = Array(Float32).new(state_size, 0.0_f32)
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  wba = WbaTrace.maybe("decode_verifier_overlap")

  # Warm exact decode and draft lane outside measured regions.
  warm_verify = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.forward_top1(weights, candidates[0], calib_count, warm_verify)
  warm_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  warm_draft_state.write(zero_state)
  ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(warm_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!

  serial_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  serial_results = [] of {Int32, Float32}
  t_serial_verify = Time.instant
  candidates.each_with_index do |tok, i|
    serial_results << ML::GGUF::Qwen35CPU.forward_top1(weights, tok, calib_count + i, serial_verify_state)
  end
  verifier_serial_ms = (Time.instant - t_serial_verify).total_milliseconds

  async_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  t_async_verify = Time.instant
  wba.try(&.point("verifier", "async_submit_begin", t_async_verify))
  candidates.each_with_index do |tok, i|
    submit_t0 = Time.instant
    sub = ML::GGUF::Qwen35CPU.forward_top1_async(weights, tok, calib_count + i, async_verify_state,
      fresh_scratch: false, scratch_namespace: "exact_verify_#{i}").not_nil!
    submit_t1 = Time.instant
    wba.try(&.mark("verifier", "async_submit_#{i}", submit_t0, submit_t1))
    submissions << sub
  end
  wait_t0 = Time.instant
  async_results = submissions.map_with_index do |sub, i|
    one_wait_t0 = Time.instant
    result = ML::GGUF::Qwen35CPU.wait_forward_top1(sub)
    one_wait_t1 = Time.instant
    wba.try(&.mark("verifier", "async_wait_#{i}", one_wait_t0, one_wait_t1))
    result
  end
  wait_t1 = Time.instant
  wba.try(&.mark("verifier", "async_wait_all", wait_t0, wait_t1))
  verifier_async_ms = (Time.instant - t_async_verify).total_milliseconds

  draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  draft_state.write(zero_state)
  t_draft = Time.instant
  serial_draft = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  draft_ms = (Time.instant - t_draft).total_milliseconds

  overlap_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  overlap_draft_state.write(zero_state)
  overlap_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_overlap = Time.instant
  wba.try(&.point("overlap", "begin", t_overlap))
  draft_submit_t0 = Time.instant
  draft_sub = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(overlap_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "lr_decode_verify_draft", command_queue_name: "lr_decode_verify_draft").not_nil!
  draft_submit_t1 = Time.instant
  wba.try(&.mark("draft", "submit", draft_submit_t0, draft_submit_t1))
  overlap_subs = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  candidates.each_with_index do |tok, i|
    submit_t0 = Time.instant
    sub = ML::GGUF::Qwen35CPU.forward_top1_async(weights, tok, calib_count + i, overlap_verify_state,
      fresh_scratch: false, scratch_namespace: "exact_overlap_#{i}").not_nil!
    submit_t1 = Time.instant
    wba.try(&.mark("verifier", "overlap_submit_#{i}", submit_t0, submit_t1))
    overlap_subs << sub
  end
  verifier_wait_t0 = Time.instant
  overlap_results = overlap_subs.map_with_index do |sub, i|
    one_wait_t0 = Time.instant
    result = ML::GGUF::Qwen35CPU.wait_forward_top1(sub)
    one_wait_t1 = Time.instant
    wba.try(&.mark("verifier", "overlap_wait_#{i}", one_wait_t0, one_wait_t1))
    result
  end
  verifier_wait_t1 = Time.instant
  wba.try(&.mark("verifier", "overlap_wait_all", verifier_wait_t0, verifier_wait_t1))
  draft_wait_t0 = Time.instant
  overlap_draft = ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(draft_sub)
  draft_wait_t1 = Time.instant
  wba.try(&.mark("draft", "wait", draft_wait_t0, draft_wait_t1))
  overlap_ms = (Time.instant - t_overlap).total_milliseconds
  wba.try(&.point("overlap", "end", Time.instant))
  wba.try(&.flush)

  max = 0.0
  serial_draft.each_with_index do |v, i|
    e = (v - overlap_draft[i]).abs.to_f64
    max = e if e > max
  end
  serial_overlap_ms = draft_ms + verifier_async_ms
  {
    steps:              n_tokens,
    draft_ms:           draft_ms,
    verifier_serial_ms: verifier_serial_ms,
    verifier_async_ms:  verifier_async_ms,
    overlap_ms:         overlap_ms,
    async_speedup:      verifier_async_ms > 0.0 ? verifier_serial_ms / verifier_async_ms : 0.0,
    overlap_speedup:    overlap_ms > 0.0 ? serial_overlap_ms / overlap_ms : 0.0,
    hidden_ms:          serial_overlap_ms - overlap_ms,
    draft_output_max:   max,
    verifier_match:     serial_results == async_results && serial_results == overlap_results,
  }
end

private def simulate_exact_verifier_ltp_proxy(weights : ML::GGUF::Qwen35Weights,
                                              token_ids : Array(Int32),
                                              calib_count : Int32) : NamedTuple(steps: Int32, decode_serial_ms: Float64, decode_queued_ms: Float64, chunk_major_ms: Float64, queued_speedup: Float64, ltp_speedup: Float64, queued_match: Bool, chunk_match: Bool)
  raise "exact verifier LTP proxy requires a non-empty held-out span" unless calib_count < token_ids.size
  hp = weights.hparams
  candidates = token_ids[calib_count, token_ids.size - calib_count]
  prefix_ids = token_ids[0, calib_count]
  max_seq = token_ids.size + candidates.size + 8

  # Warm all verifier routes outside measured regions.
  warm_decode = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.forward_top1(weights, candidates[0], calib_count, warm_decode)
  warm_chunk = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, warm_chunk)

  serial_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  serial = [] of {Int32, Float32}
  t_serial = Time.instant
  candidates.each_with_index do |tok, i|
    serial << ML::GGUF::Qwen35CPU.forward_top1(weights, tok, calib_count + i, serial_state)
  end
  decode_serial_ms = (Time.instant - t_serial).total_milliseconds

  queued_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  t_queued = Time.instant
  candidates.each_with_index do |tok, i|
    submissions << ML::GGUF::Qwen35CPU.forward_top1_async(weights, tok, calib_count + i, queued_state,
      fresh_scratch: false, scratch_namespace: "ltp_decode_#{i}").not_nil!
  end
  queued = submissions.map { |sub| ML::GGUF::Qwen35CPU.wait_forward_top1(sub) }
  decode_queued_ms = (Time.instant - t_queued).total_milliseconds

  chunk_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_chunk = Time.instant
  chunk = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, chunk_state)
  chunk_major_ms = (Time.instant - t_chunk).total_milliseconds

  serial_ids = serial.map(&.[0])
  queued_ids = queued.map(&.[0])
  chunk_ids = chunk.map(&.[0])
  {
    steps:            candidates.size,
    decode_serial_ms: decode_serial_ms,
    decode_queued_ms: decode_queued_ms,
    chunk_major_ms:   chunk_major_ms,
    queued_speedup:   decode_queued_ms > 0.0 ? decode_serial_ms / decode_queued_ms : 0.0,
    ltp_speedup:      chunk_major_ms > 0.0 ? decode_serial_ms / chunk_major_ms : 0.0,
    queued_match:     serial_ids == queued_ids,
    chunk_match:      serial_ids == chunk_ids,
  }
end

private def print_cost_truth_row(kind : String, route : String, steps : Int32, ms : Float64,
                                 plain_per_token_ms : Float64, match : Bool, note : String = "") : Nil
  per_token = steps > 0 ? ms / steps : 0.0
  rel = plain_per_token_ms > 0.0 ? per_token / plain_per_token_ms : 0.0
  tok_s = per_token > 0.0 ? 1000.0 / per_token : 0.0
  puts "cost_truth kind=#{kind} route=#{route} steps=#{steps} ms=#{ms.round(3)} ms_per_tok=#{per_token.round(3)} rel_to_plain_tok=#{rel.round(4)} tok_s=#{tok_s.round(2)} match=#{match}#{note}"
end

private def prefill_tokens_top1s_branch_split(weights : ML::GGUF::Qwen35Weights,
                                              token_ids : Array(Int32),
                                              start_pos : Int32,
                                              state : ML::GGUF::Qwen35CPU::State,
                                              guard_index : Int32) : Array({Int32, Float32})
  raise "branch split requires at least one token" if token_ids.empty?
  raise "branch split guard index out of range" unless guard_index >= 0 && guard_index < token_ids.size

  results = [] of {Int32, Float32}
  if guard_index > 0
    prefix_tokens = token_ids[0, guard_index]
    results.concat(ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, prefix_tokens, start_pos, state))
  end

  results.concat(ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, [token_ids[guard_index]], start_pos + guard_index, state))

  suffix_len = token_ids.size - guard_index - 1
  if suffix_len > 0
    suffix_tokens = token_ids[(guard_index + 1), suffix_len]
    results.concat(ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, suffix_tokens, start_pos + guard_index + 1, state))
  end

  results
end

private def prefill_tokens_top1s_branch_split_nosnap(weights : ML::GGUF::Qwen35Weights,
                                                     token_ids : Array(Int32),
                                                     start_pos : Int32,
                                                     state : ML::GGUF::Qwen35CPU::State,
                                                     guard_index : Int32) : Array({Int32, Float32})
  raise "branch split requires at least one token" if token_ids.empty?
  raise "branch split guard index out of range" unless guard_index >= 0 && guard_index < token_ids.size

  results = [] of {Int32, Float32}
  if guard_index > 0
    prefix_tokens = token_ids[0, guard_index]
    results.concat(ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, prefix_tokens, start_pos, state))
  end

  suffix_len = token_ids.size - guard_index
  if suffix_len > 0
    suffix_tokens = token_ids[guard_index, suffix_len]
    results.concat(ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, suffix_tokens, start_pos + guard_index, state))
  end

  results
end

private def simulate_self_spec_cost_truth_table(weights : ML::GGUF::Qwen35Weights,
                                                token_ids : Array(Int32),
                                                calib_count : Int32,
                                                chunk_sizes : Array(Int32),
                                                layer_bases : LayerBasisMap,
                                                rank : Int32,
                                                ffn_updown_adapters : FFNUpDownAdapterMap? = nil,
                                                draft_updown_rank : Int32? = nil,
                                                draft_updown_layer_indices : Set(Int32)? = nil,
                                                branch_split_guard_indices : Array(Int32) = [] of Int32) : Nil
  raise "cost truth table needs at least one chunk size" if chunk_sizes.empty?
  raise "cost truth table requires at least one held-out token" unless calib_count < token_ids.size
  raise "cost truth table requires Metal" unless ML::GGUF::Qwen35Metal.available?

  sizes = chunk_sizes.select { |v| v > 0 }.uniq.sort
  raise "cost truth table chunk sizes must be positive" if sizes.empty?
  heldout = token_ids.size - calib_count
  max_steps = Math.min(sizes.max, heldout)
  raise "cost truth table has no held-out tokens to measure" unless max_steps > 0

  hp = weights.hparams
  prefix_ids = token_ids[0, calib_count]
  candidates = token_ids[calib_count, max_steps]
  max_seq = token_ids.size + max_steps + 8

  warm_plain = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.forward_top1(weights, candidates[0], calib_count, warm_plain)
  warm_chunk = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, warm_chunk)

  plain_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  plain_results = [] of {Int32, Float32}
  t_plain = Time.instant
  candidates.each_with_index do |tok, i|
    plain_results << ML::GGUF::Qwen35CPU.forward_top1(weights, tok, calib_count + i, plain_state)
  end
  plain_ms = (Time.instant - t_plain).total_milliseconds
  plain_per_token = plain_ms / max_steps

  puts "cost_truth_table steps=#{max_steps} chunks=#{sizes.join(',')} layers=#{layer_bases.keys.sort.join(',')} rank=#{rank} plain_ms=#{plain_ms.round(3)} plain_ms_per_tok=#{plain_per_token.round(3)}"
  print_cost_truth_row("exact", "decode_serial", max_steps, plain_ms, plain_per_token, true, " note=autoregressive_target")

  sizes.each do |raw_k|
    k = Math.min(raw_k, max_steps)
    chunk_tokens = candidates[0, k]
    warm_k_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
    ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, chunk_tokens, calib_count, warm_k_state)
    chunk_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
    t_chunk = Time.instant
    chunk_results = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, chunk_tokens, calib_count, chunk_state)
    chunk_ms = (Time.instant - t_chunk).total_milliseconds
    match = chunk_results.map(&.[0]) == plain_results[0, k].map(&.[0])
    print_cost_truth_row("verifier", "exact_chunk_major_k#{k}", k, chunk_ms, plain_per_token, match, " note=known_candidate_span")

    unless branch_split_guard_indices.empty?
      guard_indices = if branch_split_guard_indices.includes?(-1)
                        (0...k).to_a
                      else
                        branch_split_guard_indices.select { |idx| idx >= 0 && idx < k }.uniq.sort
                      end

      guard_indices.each do |guard_index|
        warm_split_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
        prefill_tokens_top1s_branch_split(weights, chunk_tokens, calib_count, warm_split_state, guard_index)

        split_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
        t_split = Time.instant
        split_results = prefill_tokens_top1s_branch_split(weights, chunk_tokens, calib_count, split_state, guard_index)
        split_ms = (Time.instant - t_split).total_milliseconds
        split_match = split_results.map(&.[0]) == plain_results[0, k].map(&.[0])
        prefix_len = guard_index
        suffix_len = k - guard_index - 1
        split_chunks = (prefix_len > 0 ? 1 : 0) + 1 + (suffix_len > 0 ? 1 : 0)
        ratio = chunk_ms > 0.0 ? split_ms / chunk_ms : 0.0
        print_cost_truth_row("verifier_split", "branch_split_k#{k}_g#{guard_index}", k, split_ms, plain_per_token, split_match,
          " prefix=#{prefix_len} guard=1 suffix=#{suffix_len} chunks=#{split_chunks} whole_ms=#{chunk_ms.round(3)} split_over_whole=#{ratio.round(4)} note=branch_guard_verifier_shape")

        warm_nosnap_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
        prefill_tokens_top1s_branch_split_nosnap(weights, chunk_tokens, calib_count, warm_nosnap_state, guard_index)

        nosnap_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
        t_nosnap = Time.instant
        nosnap_results = prefill_tokens_top1s_branch_split_nosnap(weights, chunk_tokens, calib_count, nosnap_state, guard_index)
        nosnap_ms = (Time.instant - t_nosnap).total_milliseconds
        nosnap_match = nosnap_results.map(&.[0]) == plain_results[0, k].map(&.[0])
        nosnap_suffix_len = k - guard_index
        nosnap_chunks = (prefix_len > 0 ? 1 : 0) + (nosnap_suffix_len > 0 ? 1 : 0)
        nosnap_ratio = chunk_ms > 0.0 ? nosnap_ms / chunk_ms : 0.0
        print_cost_truth_row("verifier_split", "branch_split_nosnap_k#{k}_g#{guard_index}", k, nosnap_ms, plain_per_token, nosnap_match,
          " prefix=#{prefix_len} guard_suffix=#{nosnap_suffix_len} chunks=#{nosnap_chunks} whole_ms=#{chunk_ms.round(3)} split_over_whole=#{nosnap_ratio.round(4)} note=branch_guard_no_snapshot_runtime_shape")
      end
    end
  end

  if layer_bases.empty?
    puts "cost_truth kind=draft route=skipped steps=0 ms=0.0 ms_per_tok=0.0 rel_to_plain_tok=0.0 tok_s=0.0 match=false note=missing_lowrank_layers"
    return
  end

  state_only = simulate_self_draft_gpu_state_only_run(weights, token_ids, calib_count, max_steps, layer_bases, rank)
  print_cost_truth_row("draft_lower_bound", "lowrank_state_only_known", state_only[:steps], state_only[:chain_ms], plain_per_token, true,
    " project_ms=#{state_only[:project_ms].round(3)} note=no_lm_head_known_tokens")

  chain = simulate_self_draft_gpu_chain_run(weights, token_ids, calib_count, max_steps, layer_bases, rank)
  chain_match = chain[:agreement] == chain[:steps]
  print_cost_truth_row("draft", "lowrank_gpu_chain", chain[:steps], chain[:chain_ms], plain_per_token, chain_match,
    " agreement=#{chain[:agreement]}/#{chain[:steps]} exact_ms=#{chain[:exact_ms].round(3)} note=autoregressive_top1_id_chain")

  return unless requested_updown_rank = draft_updown_rank

  adapters = ffn_updown_adapters || raise "cost truth pca-updown requires FFN up/down adapters"
  updown_state = simulate_self_draft_gpu_state_only_run(weights, token_ids, calib_count, max_steps, layer_bases, rank,
    requested_updown_rank, adapters, draft_updown_layer_indices)
  print_cost_truth_row("draft_lower_bound", "pca_updown_state_only_known", updown_state[:steps], updown_state[:chain_ms], plain_per_token, true,
    " project_ms=#{updown_state[:project_ms].round(3)} updown_rank=#{updown_state[:updown_rank]} note=no_lm_head_known_tokens")

  updown_chain = simulate_self_draft_gpu_chain_run(weights, token_ids, calib_count, max_steps, layer_bases, rank,
    requested_updown_rank, adapters, draft_updown_layer_indices)
  updown_match = updown_chain[:agreement] == updown_chain[:steps]
  print_cost_truth_row("draft", "pca_updown_gpu_chain", updown_chain[:steps], updown_chain[:chain_ms], plain_per_token, updown_match,
    " agreement=#{updown_chain[:agreement]}/#{updown_chain[:steps]} updown_rank=#{updown_chain[:updown_rank]} exact_ms=#{updown_chain[:exact_ms].round(3)} note=autoregressive_top1_id_chain")
end

private def simulate_lowrank_draft_exact_chunk_verifier_thread_overlap(samples : Array(RecurrentSample),
                                                                       bases : BasisSet,
                                                                       weights : ML::GGUF::Qwen35Weights,
                                                                       token_ids : Array(Int32),
                                                                       lw : ML::GGUF::Qwen35RecurrentWeights,
                                                                       hp : ML::GGUF::Qwen35Hparams,
                                                                       rank : Int32,
                                                                       calib_count : Int32) : NamedTuple(steps: Int32, draft_ms: Float64, chunk_verifier_ms: Float64, serial_ms: Float64, overlap_ms: Float64, speedup: Float64, hidden_ms: Float64, draft_output_max: Float64, verifier_match: Bool)
  raise "threaded chunk verifier overlap requires Metal" unless ML::GGUF::Qwen35Metal.available?
  raise "calib_count must leave a non-empty verifier span" unless calib_count < token_ids.size
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  inputs = lowrank_layer_chunk_inputs(samples, calib_count, h_k, h_v, s, hp.n_embd)
  n_tokens = inputs[:n_tokens]
  candidates = token_ids[calib_count, n_tokens]
  prefix_ids = token_ids[0, calib_count]
  max_seq = token_ids.size + n_tokens + 8
  state_size = h_v * s * rank
  zero_state = Array(Float32).new(state_size, 0.0_f32)
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  wba = WbaTrace.maybe("chunk_verifier_thread_overlap")

  # Warm both routes outside the measured region.
  warm_verify = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, warm_verify)
  warm_draft = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  warm_draft.write(zero_state)
  ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(warm_draft, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!

  serial_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  serial_draft_state.write(zero_state)
  t_draft = Time.instant
  serial_draft = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(serial_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  draft_ms = (Time.instant - t_draft).total_milliseconds

  serial_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_verify = Time.instant
  serial_verify = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, serial_verify_state)
  chunk_verifier_ms = (Time.instant - t_verify).total_milliseconds
  serial_ms = draft_ms + chunk_verifier_ms

  thread_done = Atomic(Int32).new(0)
  thread_result = nil.as(Array({Int32, Float32})?)
  thread_error = nil.as(String?)
  overlap_draft_state = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  overlap_draft_state.write(zero_state)
  overlap_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_overlap = Time.instant
  wba.try(&.point("overlap", "begin", t_overlap))
  draft_submit_t0 = Time.instant
  draft_sub = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(overlap_draft_state, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "chunk_verifier_thread_draft", command_queue_name: "chunk_verifier_thread_draft").not_nil!
  draft_submit_t1 = Time.instant
  wba.try(&.mark("draft", "submit", draft_submit_t0, draft_submit_t1))
  Thread.new do
    begin
      STDERR.puts "chunk-thread: begin verifier" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
      thread_t0 = Time.instant
      result = ML::GGUF::Qwen35Metal::Scratch.with_namespace("chunk_verifier_thread") do
        ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, overlap_verify_state)
      end
      thread_t1 = Time.instant
      wba.try(&.mark("verifier", "thread_prefill_top1s", thread_t0, thread_t1))
      thread_result = result
      STDERR.puts "chunk-thread: verifier done" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    rescue ex
      thread_error = "#{ex.class}: #{ex.message}\n#{ex.backtrace.join('\n')}"
      STDERR.puts "chunk-thread: error #{thread_error}" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    ensure
      thread_done.set(1)
      STDERR.puts "chunk-thread: done flag set" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    end
  end
  draft_wait_t0 = Time.instant
  overlap_draft = ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(draft_sub)
  draft_wait_t1 = Time.instant
  wba.try(&.mark("draft", "wait", draft_wait_t0, draft_wait_t1))
  recv_t0 = Time.instant
  deadline = Time.instant + 120.seconds
  while thread_done.get == 0
    raise "chunk-major verifier worker did not finish within 120s" if Time.instant > deadline
    Thread.yield
  end
  if error = thread_error
    raise "chunk-major verifier worker failed: #{error}"
  end
  overlap_verify = thread_result.not_nil!
  recv_t1 = Time.instant
  wba.try(&.mark("verifier", "receive", recv_t0, recv_t1))
  overlap_ms = (Time.instant - t_overlap).total_milliseconds
  wba.try(&.point("overlap", "end", Time.instant))
  wba.try(&.flush)

  max = 0.0
  serial_draft.each_with_index do |v, i|
    e = (v - overlap_draft[i]).abs.to_f64
    max = e if e > max
  end
  {
    steps:             n_tokens,
    draft_ms:          draft_ms,
    chunk_verifier_ms: chunk_verifier_ms,
    serial_ms:         serial_ms,
    overlap_ms:        overlap_ms,
    speedup:           overlap_ms > 0.0 ? serial_ms / overlap_ms : 0.0,
    hidden_ms:         serial_ms - overlap_ms,
    draft_output_max:  max,
    verifier_match:    serial_verify.map(&.[0]) == overlap_verify.map(&.[0]),
  }
end

private def simulate_lowrank_multilayer_chunk_thread_overlap(samples : Array(RecurrentSample),
                                                             bases : BasisSet,
                                                             weights : ML::GGUF::Qwen35Weights,
                                                             token_ids : Array(Int32),
                                                             lw : ML::GGUF::Qwen35RecurrentWeights,
                                                             hp : ML::GGUF::Qwen35Hparams,
                                                             rank : Int32,
                                                             calib_count : Int32,
                                                             n_layers : Int32) : NamedTuple(steps: Int32, n_layers: Int32, draft_ms: Float64, draft_per_layer_ms: Float64, chunk_verifier_ms: Float64, serial_ms: Float64, overlap_ms: Float64, speedup: Float64, hidden_ms: Float64, draft_output_max: Float64, verifier_match: Bool)
  raise "multilayer chunk verifier overlap requires Metal" unless ML::GGUF::Qwen35Metal.available?
  raise "calib_count must leave a non-empty verifier span" unless calib_count < token_ids.size
  raise "n_layers must be positive" unless n_layers > 0
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  inputs = lowrank_layer_chunk_inputs(samples, calib_count, h_k, h_v, s, hp.n_embd)
  n_tokens = inputs[:n_tokens]
  candidates = token_ids[calib_count, n_tokens]
  prefix_ids = token_ids[0, calib_count]
  max_seq = token_ids.size + n_tokens + 8
  state_size = h_v * s * rank
  zero_state = Array(Float32).new(state_size, 0.0_f32)
  basis_buf = ML::MetalBuffer.new((h_k * rank * s).to_i64 * sizeof(Float32))
  basis_buf.write(flatten_basis_for_metal(bases, rank, h_k, s))
  wba = WbaTrace.maybe("multilayer_chunk_verifier_overlap")

  # Warm both routes outside the measured region.
  warm_verify = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, warm_verify)
  warm_draft = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
  warm_draft.write(zero_state)
  ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(warm_draft, inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
    lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
    h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!

  # Serial baseline: N layer chunks back-to-back on the default queue.
  serial_states = Array(ML::MetalBuffer).new(n_layers) do
    buf = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
    buf.write(zero_state)
    buf
  end
  last_serial_output = nil.as(Array(Float32)?)
  t_draft = Time.instant
  n_layers.times do |i|
    last_serial_output = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_buf(serial_states[i], inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
      lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
      h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale).not_nil!
  end
  draft_ms = (Time.instant - t_draft).total_milliseconds
  draft_per_layer_ms = n_layers > 0 ? draft_ms / n_layers : 0.0

  serial_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_verify = Time.instant
  serial_verify = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, serial_verify_state)
  chunk_verifier_ms = (Time.instant - t_verify).total_milliseconds
  serial_ms = draft_ms + chunk_verifier_ms

  thread_done = Atomic(Int32).new(0)
  thread_result = nil.as(Array({Int32, Float32})?)
  thread_error = nil.as(String?)
  overlap_states = Array(ML::MetalBuffer).new(n_layers) do
    buf = ML::MetalBuffer.new(state_size.to_i64 * sizeof(Float32))
    buf.write(zero_state)
    buf
  end
  overlap_verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_overlap = Time.instant
  wba.try(&.point("overlap", "begin", t_overlap))
  draft_submit_t0 = Time.instant
  draft_subs = Array(ML::GGUF::Qwen35Metal::LowRankLayerChunkSubmission).new
  n_layers.times do |i|
    sub = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_async(overlap_states[i], inputs[:inp], inputs[:q], inputs[:k], basis_buf, inputs[:v], inputs[:g], inputs[:beta], inputs[:z],
      lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
      h_k, h_v, s, rank, n_tokens, hp.rms_eps.to_f32, scale, scratch_namespace: "multi_draft_#{i}", command_queue_name: "multi_draft").not_nil!
    draft_subs << sub
  end
  draft_submit_t1 = Time.instant
  wba.try(&.mark("draft", "submit_all", draft_submit_t0, draft_submit_t1))
  Thread.new do
    begin
      STDERR.puts "multi-thread: begin verifier" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
      thread_t0 = Time.instant
      result = ML::GGUF::Qwen35Metal::Scratch.with_namespace("multi_chunk_verifier_thread") do
        ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, overlap_verify_state)
      end
      thread_t1 = Time.instant
      wba.try(&.mark("verifier", "thread_prefill_top1s", thread_t0, thread_t1))
      thread_result = result
      STDERR.puts "multi-thread: verifier done" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    rescue ex
      thread_error = "#{ex.class}: #{ex.message}\n#{ex.backtrace.join('\n')}"
      STDERR.puts "multi-thread: error #{thread_error}" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    ensure
      thread_done.set(1)
      STDERR.puts "multi-thread: done flag set" if ENV["QWEN35_CHUNK_THREAD_DEBUG"]? == "1"
    end
  end
  draft_wait_t0 = Time.instant
  last_overlap_output = nil.as(Array(Float32)?)
  draft_subs.each do |sub|
    last_overlap_output = ML::GGUF::Qwen35Metal.wait_lowrank_layer_chunk(sub)
  end
  draft_wait_t1 = Time.instant
  wba.try(&.mark("draft", "wait_all", draft_wait_t0, draft_wait_t1))
  recv_t0 = Time.instant
  deadline = Time.instant + 120.seconds
  while thread_done.get == 0
    raise "multilayer chunk-major verifier worker did not finish within 120s" if Time.instant > deadline
    Thread.yield
  end
  if error = thread_error
    raise "multilayer chunk-major verifier worker failed: #{error}"
  end
  overlap_verify = thread_result.not_nil!
  recv_t1 = Time.instant
  wba.try(&.mark("verifier", "receive", recv_t0, recv_t1))
  overlap_ms = (Time.instant - t_overlap).total_milliseconds
  wba.try(&.point("overlap", "end", Time.instant))
  wba.try(&.flush)

  max = 0.0
  if (ls = last_serial_output) && (lo = last_overlap_output)
    ls.each_with_index do |v, i|
      e = (v - lo[i]).abs.to_f64
      max = e if e > max
    end
  end
  {
    steps:              n_tokens,
    n_layers:           n_layers,
    draft_ms:           draft_ms,
    draft_per_layer_ms: draft_per_layer_ms,
    chunk_verifier_ms:  chunk_verifier_ms,
    serial_ms:          serial_ms,
    overlap_ms:         overlap_ms,
    speedup:            overlap_ms > 0.0 ? serial_ms / overlap_ms : 0.0,
    hidden_ms:          serial_ms - overlap_ms,
    draft_output_max:   max,
    verifier_match:     serial_verify.map(&.[0]) == overlap_verify.map(&.[0]),
  }
end

private def project_full_state_to_lowrank(full_state : Array(Float32),
                                          bases : Array(Array(Array(Float64))),
                                          rank : Int32,
                                          h_k : Int32, h_v : Int32, s : Int32) : Array(Float32)
  out = Array(Float32).new(h_v * s * rank, 0.0_f32)
  h_v.times do |h|
    basis = bases[h % h_k]
    r = Math.min(rank, basis.size)
    full_base = h * s * s
    out_base = h * s * rank
    s.times do |row|
      r.times do |j|
        b = basis[j]
        acc = 0.0
        s.times { |d| acc += full_state[full_base + row * s + d].to_f64 * b[d] }
        out[out_base + row * rank + j] = acc.to_f32
      end
    end
  end
  out
end

private def recurrent_layer_cpu_exact(inpSA : Array(Float32),
                                      lw : ML::GGUF::Qwen35RecurrentWeights,
                                      lstate : ML::GGUF::Qwen35CPU::LayerState,
                                      hp : ML::GGUF::Qwen35Hparams) : Array(Float32)
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  qkv_dim = 2 * h_k * s + h_v * s
  conv_k = hp.ssm_conv_kernel
  cur = ML::GGUF::Qwen35CPU.rms_norm(inpSA, lw.attn_norm, hp.rms_eps)
  proj = ML::GGUF::Qwen35CPU.qmatvec_many([lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw], cur)
  qkv_mixed = proj[0]
  z = proj[1]
  alpha = proj[2]
  beta = proj[3]
  h_v.times { |i| beta[i] = 1.0_f32 / (1.0_f32 + Math.exp(-beta[i]).to_f32) }
  ghead = Array(Float32).new(h_v) { |i| Math.exp((softplus(alpha[i] + lw.ssm_dt_bias[i]) * lw.ssm_a[i]).to_f64).to_f32 }

  conv_state = lstate.conv_state ||= Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)
  conv_out = Array(Float32).new(qkv_dim) do |ch|
    acc = 0.0_f32
    w_base = ch * conv_k
    (conv_k - 1).times { |t| acc += conv_state[t * qkv_dim + ch] * lw.ssm_conv1d[w_base + t] }
    acc + qkv_mixed[ch] * lw.ssm_conv1d[w_base + (conv_k - 1)]
  end
  (conv_k - 2).times do |t|
    src = (t + 1) * qkv_dim
    dst = t * qkv_dim
    qkv_dim.times { |ch| conv_state[dst + ch] = conv_state[src + ch] }
  end
  qkv_dim.times { |ch| conv_state[(conv_k - 2) * qkv_dim + ch] = qkv_mixed[ch] }

  silu!(conv_out)
  q_conv = Array(Float32).new(h_k * s) { |i| conv_out[i] }
  k_conv = Array(Float32).new(h_k * s) { |i| conv_out[h_k * s + i] }
  v_conv = Array(Float32).new(h_v * s) { |i| conv_out[2 * h_k * s + i] }
  h_k.times do |h|
    l2_norm_slice!(q_conv, h * s, s, hp.rms_eps)
    l2_norm_slice!(k_conv, h * s, s, hp.rms_eps)
  end

  y = Array(Float32).new(h_v * s, 0.0_f32)
  state = lstate.ssm_state ||= Array(Float32).new(h_v * s * s, 0.0_f32)
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  ML::GGUF::Qwen35CPU.delta_net_step!(state, q_conv, k_conv, v_conv, ghead, beta, y, h_k, h_v, s, scale)
  h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y, h * s, s, lw.ssm_norm, hp.rms_eps) }
  (h_v * s).times { |i| y[i] = y[i] * ML::GGUF::Qwen35CPU.silu(z[i]) }
  attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y)

  inp_l2 = Array(Float32).new(hp.n_embd) { |i| inpSA[i] + attn_out[i] }
  ffn_in = ML::GGUF::Qwen35CPU.rms_norm(inp_l2, lw.post_attention_norm, hp.rms_eps)
  gate_up = ML::GGUF::Qwen35CPU.qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in)
  gate = gate_up[0]
  up = gate_up[1]
  combined = Array(Float32).new(hp.n_ff) { |i| ML::GGUF::Qwen35CPU.silu(gate[i]) * up[i] }
  ffn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ffn_down_qw, combined)
  Array(Float32).new(hp.n_embd) { |i| inp_l2[i] + ffn_out[i] }
end

private def recurrent_layer_cpu_exact_with_ffn_activation(inpSA : Array(Float32),
                                                          lw : ML::GGUF::Qwen35RecurrentWeights,
                                                          lstate : ML::GGUF::Qwen35CPU::LayerState,
                                                          hp : ML::GGUF::Qwen35Hparams) : NamedTuple(out: Array(Float32), activation: Array(Float64), ffn_in: Array(Float64))
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  qkv_dim = 2 * h_k * s + h_v * s
  conv_k = hp.ssm_conv_kernel
  cur = ML::GGUF::Qwen35CPU.rms_norm(inpSA, lw.attn_norm, hp.rms_eps)
  proj = ML::GGUF::Qwen35CPU.qmatvec_many([lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw], cur)
  qkv_mixed = proj[0]
  z = proj[1]
  alpha = proj[2]
  beta = proj[3]
  h_v.times { |i| beta[i] = 1.0_f32 / (1.0_f32 + Math.exp(-beta[i]).to_f32) }
  ghead = Array(Float32).new(h_v) { |i| Math.exp((softplus(alpha[i] + lw.ssm_dt_bias[i]) * lw.ssm_a[i]).to_f64).to_f32 }

  conv_state = lstate.conv_state ||= Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)
  conv_out = Array(Float32).new(qkv_dim) do |ch|
    acc = 0.0_f32
    w_base = ch * conv_k
    (conv_k - 1).times { |t| acc += conv_state[t * qkv_dim + ch] * lw.ssm_conv1d[w_base + t] }
    acc + qkv_mixed[ch] * lw.ssm_conv1d[w_base + (conv_k - 1)]
  end
  (conv_k - 2).times do |t|
    src = (t + 1) * qkv_dim
    dst = t * qkv_dim
    qkv_dim.times { |ch| conv_state[dst + ch] = conv_state[src + ch] }
  end
  qkv_dim.times { |ch| conv_state[(conv_k - 2) * qkv_dim + ch] = qkv_mixed[ch] }

  silu!(conv_out)
  q_conv = Array(Float32).new(h_k * s) { |i| conv_out[i] }
  k_conv = Array(Float32).new(h_k * s) { |i| conv_out[h_k * s + i] }
  v_conv = Array(Float32).new(h_v * s) { |i| conv_out[2 * h_k * s + i] }
  h_k.times do |h|
    l2_norm_slice!(q_conv, h * s, s, hp.rms_eps)
    l2_norm_slice!(k_conv, h * s, s, hp.rms_eps)
  end

  y = Array(Float32).new(h_v * s, 0.0_f32)
  state = lstate.ssm_state ||= Array(Float32).new(h_v * s * s, 0.0_f32)
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  ML::GGUF::Qwen35CPU.delta_net_step!(state, q_conv, k_conv, v_conv, ghead, beta, y, h_k, h_v, s, scale)
  h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y, h * s, s, lw.ssm_norm, hp.rms_eps) }
  (h_v * s).times { |i| y[i] = y[i] * ML::GGUF::Qwen35CPU.silu(z[i]) }
  attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y)

  inp_l2 = Array(Float32).new(hp.n_embd) { |i| inpSA[i] + attn_out[i] }
  ffn_in = ML::GGUF::Qwen35CPU.rms_norm(inp_l2, lw.post_attention_norm, hp.rms_eps)
  gate_up = ML::GGUF::Qwen35CPU.qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in)
  gate = gate_up[0]
  up = gate_up[1]
  combined = Array(Float32).new(hp.n_ff) { |i| ML::GGUF::Qwen35CPU.silu(gate[i]) * up[i] }
  ffn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ffn_down_qw, combined)
  {
    out:        Array(Float32).new(hp.n_embd) { |i| inp_l2[i] + ffn_out[i] },
    activation: combined.map(&.to_f64),
    ffn_in:     ffn_in.map(&.to_f64),
  }
end

private def ffn_activation_vectors_for_prompt(weights : ML::GGUF::Qwen35Weights,
                                              token_ids : Array(Int32),
                                              layer_indices : Array(Int32),
                                              calib_count : Int32) : Hash(Int32, Array(Array(Float64)))
  hp = weights.hparams
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: token_ids.size + 2)
  wanted = Set(Int32).new(layer_indices)
  vectors = Hash(Int32, Array(Array(Float64))).new { |h, k| h[k] = [] of Array(Float64) }

  token_ids.each_with_index do |token_id, pos|
    x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
    weights.layers.each_with_index do |layer, il|
      case layer
      in ML::GGUF::Qwen35FullAttnWeights
        x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      in ML::GGUF::Qwen35RecurrentWeights
        if wanted.includes?(il)
          res = recurrent_layer_cpu_exact_with_ffn_activation(x, layer, state.layers[il], hp)
          vectors[il] << res[:activation] if pos < calib_count
          x = res[:out]
        else
          x = recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
        end
      end
    end
  end

  vectors
end

private def token_ids_for_prompt(tok, prompt : String, tokens_limit : Int32, repeat : Bool = true) : Array(Int32)
  token_ids = tok.encode(prompt, add_bos_override: false)
  return token_ids[0, Math.min(tokens_limit, token_ids.size)] unless repeat

  while token_ids.size < tokens_limit
    token_ids.concat(tok.encode(prompt, add_bos_override: false))
  end
  token_ids[0, tokens_limit]
end

private def lowrank_eval_layer_bases(weights : ML::GGUF::Qwen35Weights,
                                     token_ids : Array(Int32),
                                     layer_ids : Array(Int32),
                                     calib_count : Int32,
                                     max_rank : Int32,
                                     basis_mode : String,
                                     pca_iters : Int32) : NamedTuple(vectors: LayerVectorMap, bases: LayerBasisMap)
  vectors = {} of Int32 => BasisSet
  bases = {} of Int32 => BasisSet
  layer_ids.each do |il|
    layer_vectors = recurrent_k_vectors_for_prompt(weights, token_ids, il)
    vectors[il] = layer_vectors
    bases[il] = layer_vectors.map do |head_vectors|
      build_basis(head_vectors[0, calib_count], max_rank, basis_mode, pca_iters)
    end
  end
  {vectors: vectors, bases: bases}
end

private def lowrank_eval_fallback_note(fallback_threshold : Float64?,
                                       approx_steps : Int32,
                                       fallback_steps : Int32) : String
  total_steps = approx_steps + fallback_steps
  approx_rate = total_steps > 0 ? (100.0 * approx_steps / total_steps) : 0.0
  fallback_score_note = ProbeRuntime.fallback_score_mode == "raw" ? "" : " fallback_score=#{ProbeRuntime.fallback_score_mode}"
  return fallback_score_note unless fallback_threshold

  " fallback_threshold=#{fallback_threshold}#{fallback_score_note} approx_rate=#{approx_rate.round(2)}%"
end

private def run_lowrank_eval_suite(weights : ML::GGUF::Qwen35Weights,
                                   token_sets : Array(PromptTokenSet),
                                   layer_ids : Array(Int32),
                                   rank : Int32,
                                   max_rank : Int32,
                                   basis_mode : String,
                                   pca_iters : Int32,
                                   calib_tokens : Int32,
                                   generate_tokens : Int32,
                                   fallback_thresholds : Array(Float64?),
                                   refresh_interval : Int32?,
                                   oracle_refresh_interval : Int32?,
                                   output_margin_threshold : Float64?,
                                   self_spec_gammas : Array(Int32),
                                   self_spec_draft_margin : Float64?,
                                   self_spec_draft_stop_margin : Float64?,
                                   self_spec_topk_rescue : Int32?) : Nil
  token_sets.each do |token_set|
    prompt_name = token_set[:name]
    ids = token_set[:token_ids]
    prompt_calib_count = Math.min(calib_tokens, ids.size - 1)
    raise "lowrank eval suite prompt #{prompt_name.inspect} needs at least one held-out token" unless prompt_calib_count > 0 && prompt_calib_count < ids.size

    built = lowrank_eval_layer_bases(weights, ids, layer_ids, prompt_calib_count, max_rank, basis_mode, pca_iters)
    layer_bases = built[:bases]
    rank_notes = layer_ids.map { |il| "#{il}:#{basis_rank_note(layer_bases[il], rank)}" }
    puts "lowrank_eval_suite name=#{prompt_name} token_vectors=#{ids.size} calib_tokens=#{prompt_calib_count} heldout_tokens=#{ids.size - prompt_calib_count} layers=#{layer_ids.join(',')} rank=#{rank} layer_basis_effective_ranks=#{rank_notes.join(' ')}"

    fallback_thresholds.each do |fallback_threshold|
      logit = simulate_logits_policy(weights, ids, layer_bases, rank, prompt_calib_count,
        fallback_threshold, refresh_interval, oracle_refresh_interval, output_margin_threshold)
      logit_fallback_note = lowrank_eval_fallback_note(fallback_threshold, logit[:approx_steps], logit[:fallback_steps])
      output_note = output_margin_threshold ? " output_margin_threshold=#{output_margin_threshold} output_fallbacks=#{logit[:output_fallbacks]}" : ""
      refresh_note = refresh_interval ? " refresh_interval=#{refresh_interval}" : ""
      oracle_refresh_note = oracle_refresh_interval ? " oracle_refresh_interval=#{oracle_refresh_interval}" : ""
      puts "lowrank_eval_logit name=#{prompt_name} layers=#{layer_ids.join(',')} rank=#{rank} mean_cos=#{logit[:mean_cos].round(8)} min_cos=#{logit[:min_cos].round(8)} max_delta=#{logit[:max_delta].round(6)} top1_match=#{logit[:top1_match].round(2)}% top5_hit=#{logit[:top5_hit].round(2)}% mean_kl=#{logit[:mean_kl].round(8)} max_kl=#{logit[:max_kl].round(8)} min_margin=#{logit[:min_margin].round(6)} confident_mismatches=#{logit[:confident_mismatches]} approx_steps=#{logit[:approx_steps]} fallback_steps=#{logit[:fallback_steps]}#{logit_fallback_note}#{refresh_note}#{oracle_refresh_note}#{output_note}"

      next unless generate_tokens > 0

      gen = simulate_greedy_policy(weights, ids, generate_tokens, layer_bases, rank, prompt_calib_count,
        fallback_threshold, refresh_interval, oracle_refresh_interval, output_margin_threshold)
      gen_fallback_note = lowrank_eval_fallback_note(fallback_threshold, gen[:approx_steps], gen[:fallback_steps])
      gen_output_note = output_margin_threshold ? " output_margin_threshold=#{output_margin_threshold} output_fallbacks=#{gen[:output_fallbacks]}" : ""
      puts "lowrank_eval_greedy name=#{prompt_name} layers=#{layer_ids.join(',')} rank=#{rank} gen_tokens=#{generate_tokens} mean_cos=#{gen[:mean_cos].round(8)} min_cos=#{gen[:min_cos].round(8)} max_delta=#{gen[:max_delta].round(6)} top1_match=#{gen[:top1_match].round(2)}% top5_hit=#{gen[:top5_hit].round(2)}% mean_kl=#{gen[:mean_kl].round(8)} max_kl=#{gen[:max_kl].round(8)} min_margin=#{gen[:min_margin].round(6)} confident_mismatches=#{gen[:confident_mismatches]} approx_steps=#{gen[:approx_steps]} fallback_steps=#{gen[:fallback_steps]}#{gen_fallback_note}#{refresh_note}#{oracle_refresh_note}#{gen_output_note} exact_ids=#{gen[:exact_ids].join(',')} approx_ids=#{gen[:approx_ids].join(',')}"

      self_spec_gammas.each do |gamma|
        spec = simulate_self_spec_policy(weights, ids, generate_tokens, gamma, layer_bases, rank, prompt_calib_count,
          fallback_threshold, refresh_interval, nil, nil, nil, self_spec_draft_margin, self_spec_draft_stop_margin, self_spec_topk_rescue)
        spec_total_steps = spec[:approx_steps] + spec[:fallback_steps]
        spec_approx_rate = spec_total_steps > 0 ? (100.0 * spec[:approx_steps] / spec_total_steps) : 0.0
        rescue_note = self_spec_topk_rescue ? " topk_rescue=#{self_spec_topk_rescue} topk_rescues=#{spec[:topk_rescues]}" : ""
        fallback_score_note = ProbeRuntime.fallback_score_mode == "raw" ? "" : " fallback_score=#{ProbeRuntime.fallback_score_mode}"
        puts "lowrank_eval_self_spec name=#{prompt_name} layers=#{layer_ids.join(',')} rank=#{rank} gamma=#{gamma} gen_tokens=#{generate_tokens} chunks=#{spec[:chunks]} full_accept_chunks=#{spec[:full_accept_chunks]} rejections=#{spec[:rejections]}#{rescue_note} accepted_draft_tokens=#{spec[:accepted_draft_tokens]} proposed_tokens=#{spec[:proposed_tokens]} accept_rate=#{spec[:accept_rate].round(2)}% avg_accept=#{spec[:avg_accept].round(3)} verifier_tokens=#{spec[:verifier_tokens]} correction_steps=#{spec[:correction_steps]} approx_steps=#{spec[:approx_steps]} fallback_steps=#{spec[:fallback_steps]} approx_rate=#{spec_approx_rate.round(2)}%#{fallback_score_note} draft_top2_hit=#{spec[:draft_top2_hit_rate].round(2)}% draft_top5_hit=#{spec[:draft_top5_hit_rate].round(2)}% reject_top2_hits=#{spec[:reject_top2_hits]} reject_top5_hits=#{spec[:reject_top5_hits]} break_even_draft_verify_per_proposed=#{spec[:break_even_draft_verify_per_proposed].round(4)} gamma_history=#{spec[:gamma_history].join(',')} draft_min_margin_history=#{spec[:draft_min_margin_history].map { |v| v.round(4) }.join(',')} draft_low_margin_history=#{spec[:draft_low_margin_history].join(',')} exact_ids=#{spec[:exact_ids].join(',')} emitted_ids=#{spec[:emitted_ids].join(',')}"
      end
    end
  end
end

private def exact_greedy_generated_ids(weights : ML::GGUF::Qwen35Weights,
                                       prompt_ids : Array(Int32),
                                       gen_tokens : Int32) : Array(Int32)
  raise "oracle generated calibration needs a non-empty prompt" if prompt_ids.empty?
  return [] of Int32 if gen_tokens <= 0

  state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: prompt_ids.size + gen_tokens + 4)
  next_id, _ = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, prompt_ids, 0, state)
  ids = [] of Int32
  gen_tokens.times do |i|
    ids << next_id
    if i + 1 < gen_tokens
      next_id, _ = ML::GGUF::Qwen35CPU.forward_top1(weights, next_id, prompt_ids.size + i, state)
    end
  end
  ids
end

private def collect_block_residual_samples(weights : ML::GGUF::Qwen35Weights,
                                           token_ids : Array(Int32),
                                           block_start : Int32,
                                           block_end : Int32) : Array(BlockResidualSample)
  hp = weights.hparams
  raise "block start must be within layers" unless block_start >= 0 && block_start < weights.layers.size
  raise "block end must be within layers" unless block_end >= 0 && block_end < weights.layers.size
  raise "block start must be <= block end" unless block_start <= block_end

  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: token_ids.size + 2)
  samples = [] of BlockResidualSample
  token_ids.each_with_index do |token_id, pos|
    x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
    block_in = nil.as(Array(Float32)?)
    block_out = nil.as(Array(Float32)?)

    weights.layers.each_with_index do |layer, il|
      block_in = x.dup if il == block_start
      case layer
      in ML::GGUF::Qwen35FullAttnWeights
        x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      in ML::GGUF::Qwen35RecurrentWeights
        x = recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
      end
      if il == block_end
        block_out = x.dup
        break
      end
    end

    inp_vec = block_in || raise "block input was not captured"
    out_vec = block_out || raise "block output was not captured"
    samples << {
      inp:   inp_vec.map(&.to_f64),
      out:   out_vec.map(&.to_f64),
      delta: Array(Float64).new(out_vec.size) { |i| out_vec[i].to_f64 - inp_vec[i].to_f64 },
    }
  end
  samples
end

private def output_margin_impact_vectors(weights : ML::GGUF::Qwen35Weights,
                                         token_ids : Array(Int32)) : Array(Array(Float64))
  hp = weights.hparams
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: token_ids.size + 2)
  vectors = [] of Array(Float64)

  token_ids.each_with_index do |token_id, pos|
    x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
    weights.layers.each_with_index do |layer, il|
      case layer
      in ML::GGUF::Qwen35FullAttnWeights
        x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      in ML::GGUF::Qwen35RecurrentWeights
        x = recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
      end
    end

    x = ML::GGUF::Qwen35CPU.rms_norm(x, weights.output_norm, hp.rms_eps)
    logits = ML::GGUF::Qwen35CPU.qmatvec_nobias(weights.output, x)
    top2 = top_k_indices(logits, 2)
    next if top2.size < 2 || top2[0] < 0 || top2[1] < 0

    winner = ML::GGUF::Qwen35CPU.embedding_lookup(weights.output, top2[0])
    runner = ML::GGUF::Qwen35CPU.embedding_lookup(weights.output, top2[1])
    vectors << Array(Float64).new(winner.size) { |i| winner[i].to_f64 - runner[i].to_f64 }
  end

  vectors
end

private def cosine64(a : Array(Float64), b : Array(Float64)) : Float64
  dot = 0.0
  aa = 0.0
  bb = 0.0
  a.size.times do |i|
    dot += a[i] * b[i]
    aa += a[i] * a[i]
    bb += b[i] * b[i]
  end
  return 0.0 if aa <= 0.0 || bb <= 0.0

  dot / Math.sqrt(aa * bb)
end

private def block_residual_prediction_stats(samples : Array(BlockResidualSample),
                                            eval_start : Int32)
  raise "block surrogate needs held-out samples" unless eval_start < samples.size
  heldout = samples[eval_start, samples.size - eval_start]
  dim = heldout[0][:out].size
  cos_sum = 0.0
  delta_cos_sum = 0.0
  min_cos = Float64::INFINITY
  err_sq = 0.0
  delta_err_sq = 0.0
  exact_sq = 0.0
  delta_sq = 0.0
  max_delta = 0.0
  adapter_ms = 0.0

  heldout.each do |sample|
    t_pred = Time.instant
    pred_delta = yield sample[:inp]
    adapter_ms += (Time.instant - t_pred).total_milliseconds
    approx_out = Array(Float64).new(dim) { |i| sample[:inp][i] + pred_delta[i] }
    cos = cosine64(approx_out, sample[:out])
    delta_cos = cosine64(pred_delta, sample[:delta])
    cos_sum += cos
    delta_cos_sum += delta_cos
    min_cos = cos if cos < min_cos
    dim.times do |i|
      out_i = sample[:out][i]
      delta_i = sample[:delta][i]
      err = approx_out[i] - out_i
      delta_err = pred_delta[i] - delta_i
      abs_err = err.abs
      max_delta = abs_err if abs_err > max_delta
      err_sq += err * err
      delta_err_sq += delta_err * delta_err
      exact_sq += out_i * out_i
      delta_sq += delta_i * delta_i
    end
  end

  count = heldout.size
  denom = Math.max(1, count * dim)
  {
    count:           count,
    mean_cos:        cos_sum / count,
    min_cos:         min_cos,
    mean_delta_cos:  delta_cos_sum / count,
    rmse:            Math.sqrt(err_sq / denom),
    rel_rmse:        exact_sq > 0.0 ? Math.sqrt(err_sq / exact_sq) : 0.0,
    delta_rel_rmse:  delta_sq > 0.0 ? Math.sqrt(delta_err_sq / delta_sq) : 0.0,
    residual_energy: exact_sq > 0.0 ? Math.sqrt(delta_sq / exact_sq) : 0.0,
    max_delta:       max_delta,
    adapter_ms:      adapter_ms,
    adapter_ms_per_sample: adapter_ms / count,
  }
end

private def block_residual_surrogate_stats(samples : Array(BlockResidualSample),
                                           adapter : BlockResidualSurrogate,
                                           eval_start : Int32)
  block_residual_prediction_stats(samples, eval_start) do |inp|
    predict_block_residual(adapter, inp)
  end
end

private def block_residual_mixture_stats(samples : Array(BlockResidualSample),
                                         mixture : BlockResidualMixture,
                                         eval_start : Int32)
  block_residual_prediction_stats(samples, eval_start) do |inp|
    predict_block_residual(mixture, inp)
  end
end

private def block_residual_error_feedback_stats(samples : Array(BlockResidualSample),
                                                adapter : BlockResidualSurrogate | BlockResidualMixture,
                                                eval_start : Int32,
                                                decay : Float64)
  raise "block surrogate feedback decay must be in [0, 1]" unless decay >= 0.0 && decay <= 1.0
  raise "block surrogate needs held-out samples" unless eval_start < samples.size

  dim = samples[0][:out].size
  bias = Array(Float64).new(dim, 0.0)
  cos_sum = 0.0
  delta_cos_sum = 0.0
  min_cos = Float64::INFINITY
  err_sq = 0.0
  delta_err_sq = 0.0
  exact_sq = 0.0
  delta_sq = 0.0
  max_delta = 0.0
  compared = 0
  adapter_ms = 0.0

  samples.each_with_index do |sample, idx|
    t_pred = Time.instant
    pred_delta = predict_block_residual(adapter, sample[:inp])
    adapter_ms += (Time.instant - t_pred).total_milliseconds
    corrected_delta = Array(Float64).new(dim) { |i| pred_delta[i] + bias[i] }
    if idx >= eval_start
      approx_out = Array(Float64).new(dim) { |i| sample[:inp][i] + corrected_delta[i] }
      cos = cosine64(approx_out, sample[:out])
      delta_cos = cosine64(corrected_delta, sample[:delta])
      cos_sum += cos
      delta_cos_sum += delta_cos
      min_cos = cos if cos < min_cos
      dim.times do |i|
        out_i = sample[:out][i]
        delta_i = sample[:delta][i]
        err = approx_out[i] - out_i
        delta_err = corrected_delta[i] - delta_i
        abs_err = err.abs
        max_delta = abs_err if abs_err > max_delta
        err_sq += err * err
        delta_err_sq += delta_err * delta_err
        exact_sq += out_i * out_i
        delta_sq += delta_i * delta_i
      end
      compared += 1
    end

    # One-token-lag adaptive filter: after exact verification observes this
    # block residual, carry a smoothed residual-error estimate to the next token.
    dim.times { |i| bias[i] = decay * bias[i] + (1.0 - decay) * (sample[:delta][i] - pred_delta[i]) }
  end

  denom = Math.max(1, compared * dim)
  {
    count:          compared,
    mean_cos:       cos_sum / compared,
    min_cos:        min_cos,
    mean_delta_cos: delta_cos_sum / compared,
    rmse:           Math.sqrt(err_sq / denom),
    rel_rmse:       exact_sq > 0.0 ? Math.sqrt(err_sq / exact_sq) : 0.0,
    delta_rel_rmse: delta_sq > 0.0 ? Math.sqrt(delta_err_sq / delta_sq) : 0.0,
    max_delta:      max_delta,
    adapter_ms:     adapter_ms,
    adapter_ms_per_sample: adapter_ms / samples.size,
  }
end

private def ffn_activation_vectors_for_token_sets(weights : ML::GGUF::Qwen35Weights,
                                                  token_sets : Array(Array(Int32)),
                                                  layer_indices : Array(Int32),
                                                  calib_tokens : Int32) : Hash(Int32, Array(Array(Float64)))
  merged = Hash(Int32, Array(Array(Float64))).new { |h, k| h[k] = [] of Array(Float64) }
  token_sets.each do |token_ids|
    prompt_calib_count = Math.min(calib_tokens, token_ids.size)
    next if prompt_calib_count <= 0

    vectors = ffn_activation_vectors_for_prompt(weights, token_ids[0, prompt_calib_count], layer_indices, prompt_calib_count)
    vectors.each do |il, layer_vectors|
      merged[il].concat(layer_vectors)
    end
  end
  merged
end

private def percentile(values : Array(Float64), q : Float64) : Float64
  return 0.0 if values.empty?

  sorted = values.sort
  clamped = Math.max(0.0, Math.min(1.0, q))
  idx = ((sorted.size - 1).to_f64 * clamped).round.to_i
  sorted[idx]
end

private def mean(values : Array(Float64)) : Float64
  return 0.0 if values.empty?

  values.sum / values.size
end

private def ffn_block_sparsity_layer_stats(layer_id : Int32,
                                           vectors : Array(Array(Float64)),
                                           block_size : Int32) : FFNBlockSparsityLayerStats
  raise "FFN block size must be positive" unless block_size > 0
  raise "FFN block sparsity needs at least one activation vector" if vectors.empty?

  dim = vectors[0].size
  raise "FFN activation vector is empty" if dim <= 0
  blocks = (dim + block_size - 1) // block_size
  thresholds = [0.50, 0.80, 0.90, 0.95, 0.99]
  fixed_pcts = [0.05, 0.10, 0.20, 0.40]
  counts_by_threshold = Hash(Float64, Array(Float64)).new { |h, k| h[k] = [] of Float64 }
  energy_by_fixed_pct = Hash(Float64, Array(Float64)).new { |h, k| h[k] = [] of Float64 }

  vectors.each do |vec|
    raise "mixed FFN activation dimensions in layer #{layer_id}" unless vec.size == dim

    block_energy = Array(Float64).new(blocks, 0.0)
    vec.each_with_index do |value, i|
      block_energy[i // block_size] += value * value
    end
    total_energy = block_energy.sum
    sorted_energy = block_energy.sort.reverse!

    thresholds.each do |target|
      if total_energy <= 0.0
        counts_by_threshold[target] << 0.0
      else
        acc = 0.0
        needed = 0
        sorted_energy.each do |energy|
          needed += 1
          acc += energy
          break if acc >= total_energy * target
        end
        counts_by_threshold[target] << needed.to_f64
      end
    end

    fixed_pcts.each do |pct|
      keep_blocks = Math.max(1, (blocks.to_f64 * pct).ceil.to_i)
      retained = total_energy > 0.0 ? sorted_energy[0, keep_blocks].sum * 100.0 / total_energy : 0.0
      energy_by_fixed_pct[pct] << retained
    end
  end

  count_notes = thresholds.map do |target|
    values = counts_by_threshold[target]
    mean_blocks = mean(values)
    mean_read = mean_blocks * 100.0 / blocks
    "b#{(target * 100).round.to_i}=mean:#{mean_blocks.round(2)},p50:#{percentile(values, 0.50).round(2)},p90:#{percentile(values, 0.90).round(2)},read:#{mean_read.round(2)}%"
  end
  energy_notes = fixed_pcts.map do |pct|
    values = energy_by_fixed_pct[pct]
    "top#{(pct * 100).round.to_i}%=mean:#{mean(values).round(2)}%,p50:#{percentile(values, 0.50).round(2)}%,p10:#{percentile(values, 0.10).round(2)}%"
  end
  read90 = mean(counts_by_threshold[0.90]) * 100.0 / blocks
  read95 = mean(counts_by_threshold[0.95]) * 100.0 / blocks
  read99 = mean(counts_by_threshold[0.99]) * 100.0 / blocks
  puts "ffn_block_sparsity layer=#{layer_id} vectors=#{vectors.size} dim=#{dim} block_size=#{block_size} blocks=#{blocks} thresholds=#{count_notes.join(' ')} fixed_block_energy=#{energy_notes.join(' ')}"

  {
    layer:       layer_id,
    vectors:     vectors.size,
    dim:         dim,
    block_size:  block_size,
    blocks:      blocks,
    read90_mean: read90,
    read95_mean: read95,
    read99_mean: read99,
  }
end

private def print_ffn_block_sparsity_summary(stats : Array(FFNBlockSparsityLayerStats)) : Nil
  return if stats.empty?

  read90 = mean(stats.map { |row| row[:read90_mean] })
  read95 = mean(stats.map { |row| row[:read95_mean] })
  read99 = mean(stats.map { |row| row[:read99_mean] })
  verdict = if read95 <= 35.0
              "promising"
            elsif read95 <= 60.0
              "borderline"
            else
              "refute_sparse_down"
            end
  puts "ffn_block_sparsity_summary layers=#{stats.map { |row| row[:layer] }.join(',')} block_size=#{stats[0][:block_size]} read90_mean=#{read90.round(2)}% read95_mean=#{read95.round(2)}% read99_mean=#{read99.round(2)}% verdict=#{verdict}"
end

private def top_energy_block_indices(values : Array(Float64), percent : Int32, block_size : Int32) : Set(Int32)
  raise "FFN block-top percent must be in 1..100" unless percent >= 1 && percent <= 100
  raise "FFN block size must be positive" unless block_size > 0

  blocks = (values.size + block_size - 1) // block_size
  keep_blocks = Math.max(1, (blocks.to_i64 * percent + 99) // 100).to_i
  energies = Array(Float64).new(blocks, 0.0)
  values.each_with_index do |value, i|
    energies[i // block_size] += value * value
  end
  order = (0...blocks).to_a
  order.sort_by! { |block| -energies[block] }
  Set(Int32).new(order[0, keep_blocks])
end

private def retained_energy_for_blocks(values : Array(Float64), selected : Set(Int32), block_size : Int32) : Float64
  total = 0.0
  retained = 0.0
  values.each_with_index do |value, i|
    energy = value * value
    total += energy
    retained += energy if selected.includes?(i // block_size)
  end
  total > 0.0 ? retained * 100.0 / total : 0.0
end

private def nearest_ffn_input_index(train : Array(NamedTuple(ffn_in: Array(Float64), activation: Array(Float64))),
                                    query : Array(Float64)) : Int32
  best_i = 0
  best_dist = Float64::INFINITY
  train.each_with_index do |sample, i|
    x = sample[:ffn_in]
    raise "mixed FFN input dimensions in block selector" unless x.size == query.size

    dist = 0.0
    x.size.times do |d|
      delta = x[d] - query[d]
      dist += delta * delta
    end
    if dist < best_dist
      best_dist = dist
      best_i = i
    end
  end
  best_i
end

private def print_ffn_block_selector_stats(layer_id : Int32,
                                           samples : Array(FFNActivationSample),
                                           train_count : Int32,
                                           block_size : Int32,
                                           percents : Array(Int32)) : Nil
  raise "FFN block selector needs held-out samples" unless train_count > 0 && train_count < samples.size
  train = samples[0, train_count]
  eval = samples[train_count, samples.size - train_count]
  dim = samples[0][:activation].size
  blocks = (dim + block_size - 1) // block_size

  percents.each do |percent|
    pred_energy = [] of Float64
    oracle_energy = [] of Float64
    jaccard = [] of Float64
    recall = [] of Float64
    eval.each do |sample|
      nearest = train[nearest_ffn_input_index(train, sample[:ffn_in])]
      predicted = top_energy_block_indices(nearest[:activation], percent, block_size)
      oracle = top_energy_block_indices(sample[:activation], percent, block_size)
      intersection = predicted.count { |block| oracle.includes?(block) }
      union = predicted.size + oracle.size - intersection
      jaccard << (union > 0 ? intersection.to_f64 / union : 0.0)
      recall << (oracle.size > 0 ? intersection.to_f64 / oracle.size : 0.0)
      pred_energy << retained_energy_for_blocks(sample[:activation], predicted, block_size)
      oracle_energy << retained_energy_for_blocks(sample[:activation], oracle, block_size)
    end
    oracle_mean = mean(oracle_energy)
    pred_mean = mean(pred_energy)
    ratio = oracle_mean > 0.0 ? pred_mean * 100.0 / oracle_mean : 0.0
    puts "ffn_block_selector layer=#{layer_id} percent=#{percent} train=#{train.size} eval=#{eval.size} dim=#{dim} block_size=#{block_size} blocks=#{blocks} pred_energy_mean=#{pred_mean.round(2)}% pred_energy_p10=#{percentile(pred_energy, 0.10).round(2)}% oracle_energy_mean=#{oracle_mean.round(2)}% pred_oracle_ratio=#{ratio.round(2)}% jaccard_mean=#{mean(jaccard).round(4)} recall_mean=#{mean(recall).round(4)}"
  end
end

private def train_ffn_block_selector(samples : Array(FFNActivationSample),
                                     percents : Array(Int32),
                                     block_size : Int32) : FFNBlockSelector
  raise "FFN block selector needs at least one sample" if samples.empty?
  raise "FFN block size must be positive" unless block_size > 0

  by_percent = {} of Int32 => Array(Set(Int32))
  percents.uniq.each do |percent|
    by_percent[percent] = samples.map { |sample| top_energy_block_indices(sample[:activation], percent, block_size) }
  end
  FFNBlockSelector.new(samples, by_percent, block_size)
end

private def select_ffn_blocks_from_input(selector : FFNBlockSelector,
                                         ffn_in : Array(Float32),
                                         percent : Int32) : Set(Int32)
  blocks = selector.blocks_by_percent[percent]? || raise "FFN block selector missing percent #{percent}"
  query = ffn_in.map(&.to_f64)
  index = nearest_ffn_input_index(selector.samples, query)
  blocks[index]? || raise "FFN block selector index #{index} out of range"
end

private def ffn_updown_samples_for_token_sets(weights : ML::GGUF::Qwen35Weights,
                                              token_sets : Array(Array(Int32)),
                                              layer_indices : Array(Int32),
                                              calib_tokens : Int32) : Hash(Int32, Array(FFNActivationSample))
  hp = weights.hparams
  wanted = Set(Int32).new(layer_indices)
  samples = Hash(Int32, Array(FFNActivationSample)).new do |h, k|
    h[k] = [] of FFNActivationSample
  end

  token_sets.each do |token_ids|
    prompt_calib_count = Math.min(calib_tokens, token_ids.size)
    next if prompt_calib_count <= 0

    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: prompt_calib_count + 2)
    token_ids[0, prompt_calib_count].each_with_index do |token_id, pos|
      x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
      weights.layers.each_with_index do |layer, il|
        case layer
        in ML::GGUF::Qwen35FullAttnWeights
          x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
        in ML::GGUF::Qwen35RecurrentWeights
          if wanted.includes?(il)
            res = recurrent_layer_cpu_exact_with_ffn_activation(x, layer, state.layers[il], hp)
            samples[il] << {ffn_in: res[:ffn_in], activation: res[:activation]}
            x = res[:out]
          else
            x = recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
          end
        end
      end
    end
  end

  samples
end

private def solve_linear_system(a_in : Array(Array(Float64)), b_in : Array(Float64), eps : Float64 = 1.0e-12) : Array(Float64)
  n = b_in.size
  a = a_in.map(&.dup)
  b = b_in.dup
  n.times do |col|
    pivot = col
    best = a[col][col].abs
    (col + 1).upto(n - 1) do |row|
      v = a[row][col].abs
      if v > best
        best = v
        pivot = row
      end
    end
    raise "singular ridge system" if best <= eps
    if pivot != col
      a[col], a[pivot] = a[pivot], a[col]
      b[col], b[pivot] = b[pivot], b[col]
    end
    diag = a[col][col]
    col.upto(n - 1) { |j| a[col][j] /= diag }
    b[col] /= diag
    n.times do |row|
      next if row == col
      factor = a[row][col]
      next if factor.abs <= eps
      col.upto(n - 1) { |j| a[row][j] -= factor * a[col][j] }
      b[row] -= factor * b[col]
    end
  end
  b
end

private def mean_vector(vectors : Array(Array(Float64))) : Array(Float64)
  raise "need at least one vector" if vectors.empty?
  dim = vectors[0].size
  mean = Array(Float64).new(dim, 0.0)
  vectors.each do |v|
    raise "vector dimension mismatch" unless v.size == dim
    dim.times { |d| mean[d] += v[d] }
  end
  dim.times { |d| mean[d] /= vectors.size }
  mean
end

private def centered_vectors(vectors : Array(Array(Float64)), mean : Array(Float64)) : Array(Array(Float64))
  vectors.map do |v|
    raise "vector dimension mismatch" unless v.size == mean.size
    Array(Float64).new(v.size) { |d| v[d] - mean[d] }
  end
end

private def squared_distance(a : Array(Float64), b : Array(Float64)) : Float64
  raise "vector dimension mismatch" unless a.size == b.size
  acc = 0.0
  a.size.times do |i|
    d = a[i] - b[i]
    acc += d * d
  end
  acc
end

private def nearest_centroid_index(v : Array(Float64), centroids : Array(Array(Float64))) : Int32
  best = 0
  best_dist = squared_distance(v, centroids[0])
  1.upto(centroids.size - 1) do |i|
    dist = squared_distance(v, centroids[i])
    if dist < best_dist
      best = i
      best_dist = dist
    end
  end
  best.to_i32
end

private def kmeans_assignments(vectors : Array(Array(Float64)), cluster_count : Int32, iters : Int32 = 12)
  raise "need vectors for k-means" if vectors.empty?
  raise "cluster count must be positive" unless cluster_count > 0
  k = Math.min(cluster_count, vectors.size)
  dim = vectors[0].size
  centroids = Array.new(k) do |i|
    vectors[(i * vectors.size // k).clamp(0, vectors.size - 1)].dup
  end
  assignments = Array(Int32).new(vectors.size, 0)

  iters.times do
    vectors.each_with_index do |v, i|
      assignments[i] = nearest_centroid_index(v, centroids)
    end

    sums = Array.new(k) { Array(Float64).new(dim, 0.0) }
    counts = Array(Int32).new(k, 0)
    vectors.each_with_index do |v, i|
      c = assignments[i]
      counts[c] += 1
      dim.times { |d| sums[c][d] += v[d] }
    end
    k.times do |c|
      next if counts[c] == 0
      dim.times { |d| centroids[c][d] = sums[c][d] / counts[c] }
    end
  end

  {assignments: assignments, centroids: centroids}
end

private def train_block_residual_surrogate(samples : Array(BlockResidualSample),
                                           block_start : Int32,
                                           block_end : Int32,
                                           rank : Int32,
                                           pca_iters : Int32,
                                           ridge : Float64 = 1.0e-3,
                                           delta_basis_mode : String = "pca",
                                           impact_basis_seed : Array(Array(Float64))? = nil) : BlockResidualSurrogate
  raise "block surrogate rank must be positive" unless rank > 0
  raise "need block residual samples" if samples.empty?

  inputs = samples.map { |sample| sample[:inp] }
  deltas = samples.map { |sample| sample[:delta] }
  x_mean = mean_vector(inputs)
  delta_mean = mean_vector(deltas)
  centered_inputs = centered_vectors(inputs, x_mean)
  centered_deltas = centered_vectors(deltas, delta_mean)
  input_basis = pca_basis(centered_inputs, rank, pca_iters)
  delta_pca_basis = pca_basis(centered_deltas, rank, pca_iters)
  impact_basis = impact_basis_seed || [] of Array(Float64)
  delta_basis = case delta_basis_mode
                when "pca"
                  delta_pca_basis
                when "impact"
                  greedy_basis(impact_basis, rank)
                when "balanced"
                  interleaved_basis(delta_pca_basis, greedy_basis(impact_basis, rank), rank)
                else
                  raise "unsupported block surrogate delta basis #{delta_basis_mode.inspect}; expected pca, impact, or balanced"
                end
  input_rank = Math.min(rank, input_basis.size)
  delta_rank = Math.min(rank, delta_basis.size)
  raise "block surrogate needs non-empty input and delta PCA bases" unless input_rank > 0 && delta_rank > 0

  x_coeffs = Array.new(samples.size) { Array(Float64).new(input_rank, 0.0) }
  y_coeffs = Array.new(samples.size) { Array(Float64).new(delta_rank, 0.0) }
  samples.each_with_index do |sample, si|
    input_rank.times { |i| x_coeffs[si][i] = dot(centered_inputs[si], input_basis[i]) }
    delta_rank.times { |j| y_coeffs[si][j] = dot(centered_deltas[si], delta_basis[j]) }
  end

  xtx = Array.new(input_rank) { Array(Float64).new(input_rank, 0.0) }
  input_rank.times do |i|
    i.upto(input_rank - 1) do |k|
      acc = 0.0
      samples.size.times { |si| acc += x_coeffs[si][i] * x_coeffs[si][k] }
      xtx[i][k] = acc
      xtx[k][i] = acc
    end
    xtx[i][i] += ridge
  end

  coeff_weights = Array.new(input_rank) { Array(Float64).new(delta_rank, 0.0) }
  delta_rank.times do |j|
    xty = Array(Float64).new(input_rank, 0.0)
    input_rank.times do |i|
      samples.size.times { |si| xty[i] += x_coeffs[si][i] * y_coeffs[si][j] }
    end
    solution = solve_linear_system(xtx, xty)
    input_rank.times { |i| coeff_weights[i][j] = solution[i] }
  end

  BlockResidualSurrogate.new(block_start, block_end, x_mean, delta_mean,
    input_basis[0, input_rank], delta_basis[0, delta_rank], coeff_weights)
end

private def train_block_residual_mixture(samples : Array(BlockResidualSample),
                                         block_start : Int32,
                                         block_end : Int32,
                                         rank : Int32,
                                         cluster_count : Int32,
                                         pca_iters : Int32,
                                         ridge : Float64 = 1.0e-3) : BlockResidualMixture
  raise "block surrogate cluster count must be positive" unless cluster_count > 0
  global = train_block_residual_surrogate(samples, block_start, block_end, rank, pca_iters, ridge)
  return BlockResidualMixture.new([[] of Float64], [global], [samples.size], global, global.x_mean, global.input_basis) if cluster_count <= 1

  features = samples.map { |sample| block_residual_mixture_features(sample[:inp], global.x_mean, global.input_basis) }
  clustered = kmeans_assignments(features, cluster_count)
  assignments = clustered[:assignments]
  centroids = clustered[:centroids]
  groups = Array.new(centroids.size) { [] of BlockResidualSample }
  samples.each_with_index { |sample, i| groups[assignments[i]] << sample }
  adapters = [] of BlockResidualSurrogate
  cluster_sizes = [] of Int32
  groups.each do |group|
    cluster_sizes << group.size
    adapters << if group.size >= 2
                  train_block_residual_surrogate(group, block_start, block_end, rank, pca_iters, ridge)
                else
                  global
                end
  end

  BlockResidualMixture.new(centroids, adapters, cluster_sizes, global, global.x_mean, global.input_basis)
end

private def block_residual_mixture_features(inp : Array(Float64),
                                            mean : Array(Float64),
                                            basis : Array(Array(Float64))) : Array(Float64)
  basis.map do |b|
    acc = 0.0
    inp.size.times { |d| acc += (inp[d] - mean[d]) * b[d] }
    acc
  end
end

private def predict_block_residual(adapter : BlockResidualSurrogate, inp : Array(Float64)) : Array(Float64)
  raise "block surrogate input dimension mismatch" unless inp.size == adapter.x_mean.size
  input_rank = adapter.input_basis.size
  delta_rank = adapter.delta_basis.size
  x_coeff = Array(Float64).new(input_rank, 0.0)
  input_rank.times do |i|
    basis = adapter.input_basis[i]
    acc = 0.0
    inp.size.times { |d| acc += (inp[d] - adapter.x_mean[d]) * basis[d] }
    x_coeff[i] = acc
  end

  y_coeff = Array(Float64).new(delta_rank, 0.0)
  input_rank.times do |i|
    row = adapter.coeff_weights[i]
    delta_rank.times { |j| y_coeff[j] += x_coeff[i] * row[j] }
  end

  out = adapter.delta_mean.dup
  delta_rank.times do |j|
    basis = adapter.delta_basis[j]
    coeff = y_coeff[j]
    out.size.times { |d| out[d] += coeff * basis[d] }
  end
  out
end

private def predict_block_residual(mixture : BlockResidualMixture, inp : Array(Float64)) : Array(Float64)
  features = block_residual_mixture_features(inp, mixture.feature_mean, mixture.feature_basis)
  cluster = nearest_centroid_index(features, mixture.centroids)
  adapter = mixture.adapters[cluster]? || mixture.global_adapter
  predict_block_residual(adapter, inp)
end

private def logits_with_block_surrogate_policy(weights : ML::GGUF::Qwen35Weights,
                                               token_id : Int32,
                                               pos : Int32,
                                               state : ML::GGUF::Qwen35CPU::State,
                                               block_start : Int32,
                                               block_end : Int32,
                                               adapter : BlockResidualSurrogate | BlockResidualMixture,
                                               calib_count : Int32,
                                               approximate : Bool,
                                               state_mode : String = "skip") : Array(Float32)
  hp = weights.hparams
  x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
  il = 0
  while il < weights.layers.size
    if approximate && pos >= calib_count && il == block_start
      if state_mode == "shadow"
        exact_x = x
        j = block_start
        while j <= block_end
          case layer = weights.layers[j]
          in ML::GGUF::Qwen35FullAttnWeights
            exact_x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(exact_x, pos, layer, state.layers[j], hp, state.max_seq)
          in ML::GGUF::Qwen35RecurrentWeights
            exact_x = recurrent_layer_cpu_exact(exact_x, layer, state.layers[j], hp)
          end
          j += 1
        end
      elsif state_mode != "skip"
        raise "unsupported block surrogate state mode #{state_mode.inspect}; expected skip or shadow"
      end
      delta = predict_block_residual(adapter, x.map(&.to_f64))
      x = Array(Float32).new(x.size) { |d| (x[d].to_f64 + delta[d]).to_f32 }
      il = block_end + 1
      next
    end

    case layer = weights.layers[il]
    in ML::GGUF::Qwen35FullAttnWeights
      x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos, layer, state.layers[il], hp, state.max_seq)
    in ML::GGUF::Qwen35RecurrentWeights
      x = recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
    end
    il += 1
  end
  x = ML::GGUF::Qwen35CPU.rms_norm(x, weights.output_norm, hp.rms_eps)
  ML::GGUF::Qwen35CPU.qmatvec_nobias(weights.output, x)
end

private def compare_logit_rows(exact : Array(Float32),
                               approx : Array(Float32),
                               cosines : Array(Float64),
                               kls : Array(Float64)) : NamedTuple(max_delta: Float64, top1_match: Bool, top5_hit: Bool, margin: Float64, confident_mismatch: Bool)
  cosines << cosine(exact, approx)
  kls << softmax_kl(exact, approx)
  exact_top1 = top1(exact)
  approx_top1 = top1(approx)
  margin = top1_margin(exact)
  {
    max_delta:          max_abs_delta(exact, approx),
    top1_match:         exact_top1 == approx_top1,
    top5_hit:           top_k_indices(approx, 5).includes?(exact_top1),
    margin:             margin,
    confident_mismatch: exact_top1 != approx_top1 && margin >= 0.5,
  }
end

private def train_ffn_updown_adapter(samples : Array(NamedTuple(ffn_in: Array(Float64), activation: Array(Float64))),
                                     basis : Array(Array(Float64)),
                                     down_basis : Array(Array(Float32)),
                                     rank : Int32,
                                     ridge : Float64 = 1.0e-3) : FFNUpDownAdapter
  raise "need samples for FFN up/down adapter" if samples.empty?
  limit = Math.min(rank, basis.size)
  raise "need basis vectors for FFN up/down adapter" unless limit > 0
  n = samples.size
  dim = samples[0][:ffn_in].size
  x_mean = Array(Float64).new(dim, 0.0)
  samples.each do |sample|
    dim.times { |d| x_mean[d] += sample[:ffn_in][d] }
  end
  dim.times { |d| x_mean[d] /= n }

  coeffs = Array.new(n) { Array(Float64).new(limit, 0.0) }
  c_mean = Array(Float64).new(limit, 0.0)
  samples.each_with_index do |sample, si|
    limit.times do |j|
      c = dot(sample[:activation], basis[j])
      coeffs[si][j] = c
      c_mean[j] += c
    end
  end
  limit.times { |j| c_mean[j] /= n }

  gram = Array.new(n) { Array(Float64).new(n, 0.0) }
  n.times do |i|
    xi = samples[i][:ffn_in]
    i.upto(n - 1) do |k|
      xk = samples[k][:ffn_in]
      acc = 0.0
      dim.times { |d| acc += (xi[d] - x_mean[d]) * (xk[d] - x_mean[d]) }
      gram[i][k] = acc
      gram[k][i] = acc
    end
    gram[i][i] += ridge
  end

  coeff_weights = Array.new(limit) { Array(Float64).new(dim, 0.0) }
  limit.times do |j|
    y = Array(Float64).new(n) { |i| coeffs[i][j] - c_mean[j] }
    alpha = solve_linear_system(gram, y)
    w = coeff_weights[j]
    n.times do |i|
      xi = samples[i][:ffn_in]
      dim.times { |d| w[d] += alpha[i] * (xi[d] - x_mean[d]) }
    end
  end

  FFNUpDownAdapter.new(x_mean, c_mean, coeff_weights, down_basis[0, limit])
end

private def recurrent_layer_cpu_lowrank(inpSA : Array(Float32),
                                        lw : ML::GGUF::Qwen35RecurrentWeights,
                                        lstate : ML::GGUF::Qwen35CPU::LayerState,
                                        hp : ML::GGUF::Qwen35Hparams,
                                        bases : BasisSet,
                                        rank : Int32,
                                        lr_state : LowRankState,
                                        fallback_threshold : Float64? = nil,
                                        force_fallback : Bool = false,
                                        use_metal_lowrank : Bool = false,
                                        project_coeffs_on_gpu : Bool = false,
                                        use_metal_layer_updown : Bool = false,
                                        draft_variant : String = "lowrank",
                                        ffn_basis : Array(Array(Float64))? = nil,
                                        ffn_adapter : FFNAdapter? = nil,
                                        ffn_updown_adapter : FFNUpDownAdapter? = nil,
                                        ffn_block_selector : FFNBlockSelector? = nil) : Array(Float32)
  return inpSA if draft_variant == "skip-layer"

  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  qkv_dim = 2 * h_k * s + h_v * s
  conv_k = hp.ssm_conv_kernel

  unless lr_state.initialized
    full_state = lstate.ssm_state ||= Array(Float32).new(h_v * s * s, 0.0_f32)
    lr_state.m = project_full_state_to_lowrank(full_state, bases, rank, h_k, h_v, s)
    lr_state.full_state_current = true
    lr_state.initialized = true
  end

  cur = ML::GGUF::Qwen35CPU.rms_norm(inpSA, lw.attn_norm, hp.rms_eps)
  proj = ML::GGUF::Qwen35CPU.qmatvec_many([lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw], cur)
  qkv_mixed = proj[0]
  z = proj[1]
  alpha = proj[2]
  beta = proj[3]
  h_v.times { |i| beta[i] = 1.0_f32 / (1.0_f32 + Math.exp(-beta[i]).to_f32) }
  ghead = Array(Float32).new(h_v) { |i| Math.exp((softplus(alpha[i] + lw.ssm_dt_bias[i]) * lw.ssm_a[i]).to_f64).to_f32 }

  conv_state = lstate.conv_state ||= Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)
  conv_out = Array(Float32).new(qkv_dim) do |ch|
    acc = 0.0_f32
    w_base = ch * conv_k
    (conv_k - 1).times { |t| acc += conv_state[t * qkv_dim + ch] * lw.ssm_conv1d[w_base + t] }
    acc + qkv_mixed[ch] * lw.ssm_conv1d[w_base + (conv_k - 1)]
  end
  (conv_k - 2).times do |t|
    src = (t + 1) * qkv_dim
    dst = t * qkv_dim
    qkv_dim.times { |ch| conv_state[dst + ch] = conv_state[src + ch] }
  end
  qkv_dim.times { |ch| conv_state[(conv_k - 2) * qkv_dim + ch] = qkv_mixed[ch] }

  silu!(conv_out)
  q_conv = Array(Float32).new(h_k * s) { |i| conv_out[i] }
  k_conv = Array(Float32).new(h_k * s) { |i| conv_out[h_k * s + i] }
  v_conv = Array(Float32).new(h_v * s) { |i| conv_out[2 * h_k * s + i] }
  h_k.times do |h|
    l2_norm_slice!(q_conv, h * s, s, hp.rms_eps)
    l2_norm_slice!(k_conv, h * s, s, hp.rms_eps)
  end

  y = Array(Float32).new(h_v * s, 0.0_f32)
  scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
  fallback = force_fallback
  if threshold = fallback_threshold
    fallback ||= max_k_residual_score(k_conv, ghead, beta, bases, rank, h_k, s, ProbeRuntime.fallback_score_mode) > threshold
  end
  routed_out = nil.as(Array(Float32)?)
  if fallback
    sync_lowrank_state_from_metal!(lr_state) if use_metal_lowrank
    unless lr_state.full_state_current
      lstate.ssm_state = reconstruct_lowrank_state(lr_state.m, bases, rank, h_k, h_v, s)
    end
    state = lstate.ssm_state.not_nil!
    ML::GGUF::Qwen35CPU.delta_net_step!(state, q_conv, k_conv, v_conv, ghead, beta, y, h_k, h_v, s, scale)
    lr_state.m = project_full_state_to_lowrank(state, bases, rank, h_k, h_v, s)
    lr_state.full_state_current = true
    lr_state.fallback_steps += 1
  else
    sample = RecurrentSample.new(q_conv, k_conv, v_conv, ghead, beta)
    pca_updown_rank = draft_variant_ffn_pca_updown_rank(draft_variant)
    if pca_updown_rank && use_metal_layer_updown && use_metal_lowrank && project_coeffs_on_gpu
      adapter = ffn_updown_adapter || raise "draft variant #{draft_variant.inspect} requires FFN up/down adapter"
      state_buf = lowrank_state_buffer!(lr_state)
      basis_buf = lowrank_basis_buffer!(lr_state, bases, rank, h_k, s)
      updown = updown_adapter_buffers!(lr_state, adapter, pca_updown_rank, hp.n_embd)
      out = ML::GGUF::Qwen35Metal.lowrank_delta_chunk_projected_layer_updown_buf(
        state_buf, inpSA, q_conv, k_conv, basis_buf, v_conv, ghead, beta, z,
        lw.ssm_norm, lw.ssm_out_qw, lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
        updown[:x_mean], updown[:c_mean], updown[:coeff_w], updown[:down],
        h_k, h_v, s, rank, 1, updown[:rank], hp.rms_eps.to_f32, scale
      ).not_nil!
      lr_state.full_state_current = false
      lr_state.approx_steps += 1
      routed_out = out
    else
      if use_metal_lowrank
        lowrank_projected_delta_step_metal!(lr_state, sample, bases, rank, y, h_k, h_v, s, scale, project_coeffs_on_gpu)
      else
        lowrank_projected_delta_step!(lr_state.m, sample, bases, rank, y, h_k, h_v, s, scale)
      end
      lr_state.full_state_current = false
      lr_state.approx_steps += 1
    end
  end
  return routed_out.not_nil! if routed_out

  h_v.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(y, h * s, s, lw.ssm_norm, hp.rms_eps) }
  (h_v * s).times { |i| y[i] = y[i] * ML::GGUF::Qwen35CPU.silu(z[i]) }
  attn_out = ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ssm_out_qw, y)

  inp_l2 = Array(Float32).new(hp.n_embd) { |i| inpSA[i] + attn_out[i] }
  return inp_l2 if draft_variant == "lowrank-no-ffn"

  ffn_in = ML::GGUF::Qwen35CPU.rms_norm(inp_l2, lw.post_attention_norm, hp.rms_eps)
  blockpred_blocks = nil.as(Set(Int32)?)
  if percent = draft_variant_ffn_block_pred_percent(draft_variant)
    selector = ffn_block_selector || raise "draft variant #{draft_variant.inspect} requires FFN block selector"
    blockpred_blocks = select_ffn_blocks_from_input(selector, ffn_in, percent)
  end
  if pca_updown_rank = draft_variant_ffn_pca_updown_rank(draft_variant)
    adapter = ffn_updown_adapter || raise "draft variant #{draft_variant.inspect} requires FFN up/down adapter"
    ffn_out = ffn_out_from_updown_adapter(ffn_in, adapter, pca_updown_rank)
    return Array(Float32).new(hp.n_embd) { |i| inp_l2[i] + ffn_out[i] }
  end

  gate_up = ML::GGUF::Qwen35CPU.qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], ffn_in)
  gate = gate_up[0]
  up = gate_up[1]
  combined = Array(Float32).new(hp.n_ff) { |i| ML::GGUF::Qwen35CPU.silu(gate[i]) * up[i] }
  if percent = draft_variant_ffn_top_percent(draft_variant)
    keep_top_abs_percent!(combined, percent)
  end
  if percent = draft_variant_ffn_block_top_percent(draft_variant)
    keep_top_energy_blocks!(combined, percent, DEFAULT_FFN_SPARSE_BLOCK_SIZE)
  end
  if selected = blockpred_blocks
    selector = ffn_block_selector || raise "draft variant #{draft_variant.inspect} requires FFN block selector"
    zero_except_blocks!(combined, selected, selector.block_size)
  end
  if pca_rank = draft_variant_ffn_pca_rank(draft_variant)
    basis = ffn_basis || raise "draft variant #{draft_variant.inspect} requires FFN activation basis"
    project_vector_with_basis!(combined, basis, pca_rank)
  end
  ffn_out = if pca_down_rank = draft_variant_ffn_pca_down_rank(draft_variant)
              adapter = ffn_adapter || raise "draft variant #{draft_variant.inspect} requires FFN down adapter"
              ffn_down_from_adapter(combined, adapter, pca_down_rank)
            else
              ML::GGUF::Qwen35CPU.qmatvec_nobias(lw.ffn_down_qw, combined)
            end
  Array(Float32).new(hp.n_embd) { |i| inp_l2[i] + ffn_out[i] }
end

private def logits_with_target_layer(weights : ML::GGUF::Qwen35Weights,
                                     token_id : Int32,
                                     pos : Int32,
                                     state : ML::GGUF::Qwen35CPU::State,
                                     target_layer : Int32,
                                     bases : Array(Array(Array(Float64))),
                                     rank : Int32,
                                     calib_count : Int32,
                                     lr_state : LowRankState?,
                                     approximate : Bool) : Array(Float32)
  hp = weights.hparams
  x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
  weights.layers.each_with_index do |layer, il|
    case layer
    in ML::GGUF::Qwen35FullAttnWeights
      x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos, layer, state.layers[il], hp, state.max_seq)
    in ML::GGUF::Qwen35RecurrentWeights
      if il == target_layer
        x = if approximate && pos >= calib_count
              recurrent_layer_cpu_lowrank(x, layer, state.layers[il], hp, bases, rank, lr_state.not_nil!)
            else
              recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
            end
      else
        x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos, layer, state.layers[il], hp, state.max_seq)
      end
    end
  end
  x = ML::GGUF::Qwen35CPU.rms_norm(x, weights.output_norm, hp.rms_eps)
  ML::GGUF::Qwen35CPU.qmatvec_nobias(weights.output, x)
end

private def logits_with_lowrank_policy(weights : ML::GGUF::Qwen35Weights,
                                       token_id : Int32,
                                       pos : Int32,
                                       state : ML::GGUF::Qwen35CPU::State,
                                       layer_bases : LayerBasisMap,
                                       rank : Int32,
                                       calib_count : Int32,
                                       lr_states : Hash(Int32, LowRankState),
                                       fallback_threshold : Float64?,
                                       refresh_interval : Int32?,
                                       approximate : Bool,
                                       use_metal_lowrank : Bool = false,
                                       project_coeffs_on_gpu : Bool = false,
                                       use_metal_layer_updown : Bool = false,
                                       draft_variant : String = "lowrank",
                                       ffn_bases : FFNBasisMap? = nil,
                                       ffn_adapters : FFNAdapterMap? = nil,
                                       ffn_updown_adapters : FFNUpDownAdapterMap? = nil,
                                       ffn_block_selectors : FFNBlockSelectorMap? = nil) : Array(Float32)
  hp = weights.hparams
  x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)
  early_exit_layers = approximate ? cheap_draft_early_exit_layers(draft_variant) : nil
  weights.layers.each_with_index do |layer, il|
    case layer
    in ML::GGUF::Qwen35FullAttnWeights
      x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos, layer, state.layers[il], hp, state.max_seq)
    in ML::GGUF::Qwen35RecurrentWeights
      if bases = layer_bases[il]?
        x = if approximate && pos >= calib_count
              lr_state = lr_states[il] ||= LowRankState.new
              force_refresh = if interval = refresh_interval
                                interval > 0 && ((pos - calib_count) % interval == 0)
                              else
                                false
                              end
              ffn_basis = ffn_bases ? ffn_bases.not_nil![il]? : nil
              ffn_adapter = ffn_adapters ? ffn_adapters.not_nil![il]? : nil
              ffn_updown_adapter = ffn_updown_adapters ? ffn_updown_adapters.not_nil![il]? : nil
              ffn_block_selector = ffn_block_selectors ? ffn_block_selectors.not_nil![il]? : nil
              recurrent_layer_cpu_lowrank(x, layer, state.layers[il], hp, bases, rank, lr_state, fallback_threshold, force_refresh, use_metal_lowrank, project_coeffs_on_gpu, use_metal_layer_updown, draft_variant, ffn_basis, ffn_adapter, ffn_updown_adapter, ffn_block_selector)
            else
              recurrent_layer_cpu_exact(x, layer, state.layers[il], hp)
            end
      else
        x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos, layer, state.layers[il], hp, state.max_seq)
      end
    end
    break if early_exit_layers && (il + 1) >= early_exit_layers
  end
  x = ML::GGUF::Qwen35CPU.rms_norm(x, weights.output_norm, hp.rms_eps)
  ML::GGUF::Qwen35CPU.qmatvec_nobias(weights.output, x)
end

private def refresh_due?(pos : Int32, calib_count : Int32, interval : Int32?) : Bool
  return false unless n = interval
  return false unless n > 0 && pos >= calib_count

  ((pos - calib_count + 1) % n) == 0
end

private def sync_lowrank_shadow!(approx_state : ML::GGUF::Qwen35CPU::State,
                                 exact_state : ML::GGUF::Qwen35CPU::State,
                                 layer_bases : LayerBasisMap,
                                 lr_states : Hash(Int32, LowRankState),
                                 rank : Int32,
                                 hp : ML::GGUF::Qwen35Hparams) : Nil
  approx_state.copy_from!(exact_state)
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  layer_bases.each do |il, bases|
    next unless state = approx_state.layers[il].ssm_state

    lr_state = lr_states[il] ||= LowRankState.new
    lr_state.m = project_full_state_to_lowrank(state, bases, rank, h_k, h_v, s)
    lr_state.full_state_current = true
    lr_state.initialized = true
  end
end

private def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0
  aa = 0.0
  bb = 0.0
  a.size.times do |i|
    av = a[i].to_f64
    bv = b[i].to_f64
    dot += av * bv
    aa += av * av
    bb += bv * bv
  end
  dot / Math.sqrt(aa * bb)
end

private def top1(v : Array(Float32)) : Int32
  best = 0
  best_v = v[0]
  v.each_with_index do |x, i|
    if x > best_v
      best = i
      best_v = x
    end
  end
  best.to_i32
end

private def planned_gpu_update_risk_offsets(weights : ML::GGUF::Qwen35Weights,
                                            prompt_ids : Array(Int32),
                                            gen_tokens : Int32,
                                            layer_bases : LayerBasisMap,
                                            rank : Int32,
                                            calib_count : Int32,
                                            fallback_threshold : Float64) : NamedTuple(offsets: Array(Int32), layer_offsets: Hash(Int32, Array(Int32)), approx_steps: Int32, fallback_steps: Int32, exact_ids: Array(Int32))
  return {offsets: [] of Int32, layer_offsets: {} of Int32 => Array(Int32), approx_steps: 0, fallback_steps: 0, exact_ids: [] of Int32} if prompt_ids.empty? || gen_tokens <= 0 || layer_bases.empty?

  hp = weights.hparams
  max_seq = prompt_ids.size + gen_tokens + 8
  prefix_ids = prompt_ids[0, prompt_ids.size - 1]
  prompt_last_token = prompt_ids[-1]
  prompt_pos_last = prompt_ids.size - 1

  exact_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  draft_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  draft_lr_states = {} of Int32 => LowRankState
  sync_lowrank_shadow!(draft_state, exact_state, layer_bases, draft_lr_states, rank, hp)

  offsets = [] of Int32
  layer_offsets = {} of Int32 => Array(Int32)
  exact_ids = [] of Int32
  token = prompt_last_token

  gen_tokens.times do |offset|
    pos = prompt_pos_last + offset
    exact_token = ML::GGUF::Qwen35CPU.forward_top1(weights, token, pos.to_i32, exact_state)[0]
    exact_ids << exact_token

    fallback_before = {} of Int32 => Int32
    draft_lr_states.each { |il, lr| fallback_before[il] = lr.fallback_steps }
    logits_with_lowrank_policy(weights, token, pos.to_i32, draft_state,
      layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, nil, true)
    any_fallback = false
    draft_lr_states.each do |il, lr|
      if lr.fallback_steps > (fallback_before[il]? || 0)
        any_fallback = true
        (layer_offsets[il] ||= [] of Int32) << offset
      end
    end
    offsets << offset if any_fallback

    token = exact_token
  end

  layer_offsets.each { |_, values| values.uniq!.sort! }

  {
    offsets:        offsets.uniq.sort,
    layer_offsets:  layer_offsets,
    approx_steps:   draft_lr_states.values.sum(&.approx_steps),
    fallback_steps: draft_lr_states.values.sum(&.fallback_steps),
    exact_ids:      exact_ids,
  }
end

private def format_layer_offsets(layer_offsets : Hash(Int32, Array(Int32))) : String
  layer_offsets.keys.sort.map { |il| "#{il}:#{layer_offsets[il].join(',')}" }.join(";")
end

private def top_k_indices(v : Array(Float32), k : Int32) : Array(Int32)
  best_i = Array(Int32).new(k, -1)
  best_v = Array(Float32).new(k, -Float32::INFINITY)
  v.each_with_index do |x, i|
    next if x <= best_v[-1]

    slot = k - 1
    while slot > 0 && x > best_v[slot - 1]
      best_v[slot] = best_v[slot - 1]
      best_i[slot] = best_i[slot - 1]
      slot -= 1
    end
    best_v[slot] = x
    best_i[slot] = i.to_i32
  end
  best_i
end

private def hidden_row_norm(hidden : Array(Float32), row : Int32, dim : Int32) : Float64
  base = row * dim
  acc = 0.0
  dim.times do |i|
    x = hidden[base + i].to_f64
    acc += x * x
  end
  Math.sqrt(acc)
end

private def hidden_row_cosine(hidden : Array(Float32),
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

private def current_hidden_nearest_labels(hidden : Array(Float32),
                                          labels : Array(Int32),
                                          norms : Array(Float64),
                                          dim : Int32,
                                          eval_row : Int32,
                                          train_count : Int32,
                                          top_k : Int32) : NamedTuple(ids: Array(Int32), best_cos: Float64)
  by_label = {} of Int32 => Float64
  train_count.times do |j|
    sim = hidden_row_cosine(hidden, dim, eval_row, j, norms)
    label = labels[j]
    prev = by_label[label]?
    by_label[label] = sim if prev.nil? || sim > prev
  end

  ranked = by_label.to_a.sort_by { |pair| -pair[1] }
  {
    ids:      ranked.first(top_k).map { |pair| pair[0] },
    best_cos: ranked.empty? ? -Float64::INFINITY : ranked[0][1],
  }
end

private def current_hidden_label_centroids(hidden : Array(Float32),
                                           labels : Array(Int32),
                                           dim : Int32,
                                           train_count : Int32) : NamedTuple(centroids: Hash(Int32, Array(Float64)), norms: Hash(Int32, Float64), counts: Hash(Int32, Int32))
  sums = {} of Int32 => Array(Float64)
  counts = Hash(Int32, Int32).new(0)
  train_count.times do |row|
    label = labels[row]
    sum = sums[label] ||= Array(Float64).new(dim, 0.0)
    base = row * dim
    dim.times { |i| sum[i] += hidden[base + i].to_f64 }
    counts[label] += 1
  end

  centroids = {} of Int32 => Array(Float64)
  norms = {} of Int32 => Float64
  sums.each do |label, sum|
    inv = 1.0 / counts[label]
    centroid = Array(Float64).new(dim) { |i| sum[i] * inv }
    centroids[label] = centroid
    norms[label] = Math.sqrt(dot(centroid, centroid))
  end

  {centroids: centroids, norms: norms, counts: counts}
end

private def hidden_row_centroid_cosine(hidden : Array(Float32),
                                       dim : Int32,
                                       row : Int32,
                                       row_norm : Float64,
                                       centroid : Array(Float64),
                                       centroid_norm : Float64) : Float64
  denom = row_norm * centroid_norm
  return -Float64::INFINITY if denom <= 0.0

  base = row * dim
  acc = 0.0
  dim.times { |i| acc += hidden[base + i].to_f64 * centroid[i] }
  acc / denom
end

private def hidden_vector_norm(v : Array(Float32)) : Float64
  acc = 0.0
  v.each do |x|
    xf = x.to_f64
    acc += xf * xf
  end
  Math.sqrt(acc)
end

private def hidden_vector_row_cosine(v : Array(Float32),
                                     v_norm : Float64,
                                     hidden : Array(Float32),
                                     dim : Int32,
                                     row : Int32,
                                     row_norms : Array(Float64)) : Float64
  denom = v_norm * row_norms[row]
  return -Float64::INFINITY if denom <= 0.0

  base = row * dim
  acc = 0.0
  dim.times { |i| acc += v[i].to_f64 * hidden[base + i].to_f64 }
  acc / denom
end

private def current_hidden_nearest_labels_for_vector(v : Array(Float32),
                                                     hidden : Array(Float32),
                                                     labels : Array(Int32),
                                                     row_norms : Array(Float64),
                                                     dim : Int32,
                                                     train_count : Int32,
                                                     top_k : Int32) : NamedTuple(ids: Array(Int32), best_row: Int32, best_cos: Float64)
  v_norm = hidden_vector_norm(v)
  by_label = {} of Int32 => Float64
  best_row = 0
  best_cos = -Float64::INFINITY
  train_count.times do |j|
    sim = hidden_vector_row_cosine(v, v_norm, hidden, dim, j, row_norms)
    if sim > best_cos
      best_cos = sim
      best_row = j
    end
    label = labels[j]
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

private def current_hidden_centroid_labels(hidden : Array(Float32),
                                           norms : Array(Float64),
                                           dim : Int32,
                                           eval_row : Int32,
                                           centroid_pack : NamedTuple(centroids: Hash(Int32, Array(Float64)), norms: Hash(Int32, Float64), counts: Hash(Int32, Int32)),
                                           top_k : Int32) : NamedTuple(ids: Array(Int32), best_cos: Float64)
  ranked = centroid_pack[:centroids].map do |label, centroid|
    sim = hidden_row_centroid_cosine(hidden, dim, eval_row, norms[eval_row], centroid, centroid_pack[:norms][label])
    {label, sim}
  end
  ranked.sort_by! { |pair| -pair[1] }
  {
    ids:      ranked.first(top_k).map { |pair| pair[0] },
    best_cos: ranked.empty? ? -Float64::INFINITY : ranked[0][1],
  }
end

private def run_current_hidden_generate_proposal(weights : ML::GGUF::Qwen35Weights,
                                                 prompt_name : String,
                                                 prompt_ids : Array(Int32),
                                                 gen_tokens : Int32,
                                                 top_k : Int32) : NamedTuple(eval_samples: Int32, top_k: Int32, top1_hits: Int32, topk_hits: Int32, transition_hits: Int32, collect_ms: Float64, proposal_ms: Float64, avg_best_cos: Float64, top1_rate: Float64, topk_rate: Float64, transition_rate: Float64, exact_ids: Array(Int32))
  raise "current-hidden generate proposal needs positive gen_tokens" unless gen_tokens > 0
  raise "current-hidden generate proposal needs a non-empty prompt" if prompt_ids.empty?

  hp = weights.hparams
  max_seq = prompt_ids.size + gen_tokens + 4
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  t_collect = Time.instant
  pair = ML::GGUF::Qwen35CPU.prefill_tokens_hidden_top1s(weights, prompt_ids, 0, state)
  collect_ms = (Time.instant - t_collect).total_milliseconds

  hidden = pair[:hidden]
  labels = pair[:top1s].map { |row| row[0] }
  rows = labels.size
  dim = hp.n_embd
  norms = Array(Float64).new(rows) { |i| hidden_row_norm(hidden, i, dim) }

  current_hidden = hidden[(rows - 1) * dim, dim]
  top1_hits = 0
  topk_hits = 0
  transition_hits = 0
  exact_ids = [] of Int32
  best_cosines = [] of Float64

  t_probe = Time.instant
  gen_tokens.times do |step|
    exact_id = ML::GGUF::Qwen35CPU.hidden_top1(weights, current_hidden)[0]
    exact_ids << exact_id
    proposal = current_hidden_nearest_labels_for_vector(current_hidden, hidden, labels, norms, dim, rows, top_k)
    ids = proposal[:ids]
    top1_hits += 1 if ids[0]? == exact_id
    topk_hits += 1 if ids.includes?(exact_id)
    nearest = proposal[:best_row]
    if nearest + 1 < rows && labels[nearest + 1] == exact_id
      transition_hits += 1
    end
    best_cosines << proposal[:best_cos]
    break if step == gen_tokens - 1
    current_hidden = ML::GGUF::Qwen35CPU.forward_hidden(weights, exact_id, prompt_ids.size + step, state)
  end
  proposal_ms = (Time.instant - t_probe).total_milliseconds

  eval_samples = exact_ids.size
  {
    eval_samples:    eval_samples,
    top_k:           top_k,
    top1_hits:       top1_hits,
    topk_hits:       topk_hits,
    transition_hits: transition_hits,
    collect_ms:      collect_ms,
    proposal_ms:     proposal_ms,
    avg_best_cos:    best_cosines.empty? ? 0.0 : best_cosines.sum / best_cosines.size,
    top1_rate:       eval_samples > 0 ? 100.0 * top1_hits / eval_samples : 0.0,
    topk_rate:       eval_samples > 0 ? 100.0 * topk_hits / eval_samples : 0.0,
    transition_rate: eval_samples > 0 ? 100.0 * transition_hits / eval_samples : 0.0,
    exact_ids:       exact_ids,
  }
end

private def hidden_row_with_delta(hidden : Array(Float32),
                                  dim : Int32,
                                  src_row : Int32,
                                  from_row : Int32,
                                  to_row : Int32) : Array(Float32)
  src_base = src_row * dim
  from_base = from_row * dim
  to_base = to_row * dim
  Array(Float32).new(dim) { |i| hidden[src_base + i] + hidden[to_base + i] - hidden[from_base + i] }
end

private def hidden_row64(hidden : Array(Float32), row : Int32, dim : Int32) : Array(Float64)
  base = row * dim
  Array(Float64).new(dim) { |i| hidden[base + i].to_f64 }
end

private def train_current_hidden_pca_transition(hidden : Array(Float32),
                                                dim : Int32,
                                                train_count : Int32,
                                                rank : Int32,
                                                pca_iters : Int32) : BlockResidualSurrogate?
  return nil if rank <= 0 || train_count < 3

  samples = [] of BlockResidualSample
  (0...(train_count - 1)).each do |row|
    inp = hidden_row64(hidden, row, dim)
    out = hidden_row64(hidden, row + 1, dim)
    delta = Array(Float64).new(dim) { |i| out[i] - inp[i] }
    samples << {inp: inp, out: out, delta: delta}
  end

  train_rank = Math.min(rank, samples.size - 1)
  return nil if train_rank <= 0
  train_block_residual_surrogate(samples, -1, -1, train_rank, pca_iters)
rescue ex
  nil
end

private def run_current_hidden_proposal(weights : ML::GGUF::Qwen35Weights,
                                        prompt_name : String,
                                        token_ids : Array(Int32),
                                        calib_count : Int32,
                                        top_k : Int32,
                                        pca_transition_rank : Int32,
                                        pca_iters : Int32) : NamedTuple(eval_samples: Int32, train_samples: Int32, top_k: Int32, top1_hits: Int32, topk_hits: Int32, centroid_top1_hits: Int32, centroid_topk_hits: Int32, unique_train_labels: Int32, collect_ms: Float64, proposal_ms: Float64, avg_best_cos: Float64, p50_best_cos: Float64, min_best_cos: Float64, centroid_avg_best_cos: Float64, top1_rate: Float64, topk_rate: Float64, centroid_top1_rate: Float64, centroid_topk_rate: Float64, transition_samples: Int32, transition_label_hits: Int32, transition_delta_hits: Int32, pca_transition_samples: Int32, pca_transition_hits: Int32, transition_label_rate: Float64, transition_delta_rate: Float64, pca_transition_rate: Float64, transition_ms: Float64, pca_transition_ms: Float64, pca_transition_effective_rank: Int32)
  raise "current-hidden proposal top_k must be positive" unless top_k > 0
  raise "current-hidden proposal needs at least one train and one eval token" unless calib_count > 0 && calib_count < token_ids.size

  hp = weights.hparams
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: token_ids.size + 2)
  t_collect = Time.instant
  pair = ML::GGUF::Qwen35CPU.prefill_tokens_hidden_top1s(weights, token_ids, 0, state)
  collect_ms = (Time.instant - t_collect).total_milliseconds

  hidden = pair[:hidden]
  labels = pair[:top1s].map { |row| row[0] }
  rows = labels.size
  train_count = calib_count.clamp(1, rows - 1)
  eval_count = rows - train_count
  raise "current-hidden proposal has no eval rows for #{prompt_name}" unless eval_count > 0

  dim = hp.n_embd
  norms = Array(Float64).new(rows) { |i| hidden_row_norm(hidden, i, dim) }
  centroid_pack = current_hidden_label_centroids(hidden, labels, dim, train_count)
  top1_hits = 0
  topk_hits = 0
  centroid_top1_hits = 0
  centroid_topk_hits = 0
  best_cosines = [] of Float64
  centroid_best_cosines = [] of Float64
  nearest_rows = [] of Int32

  t_probe = Time.instant
  (train_count...rows).each do |row|
    proposal = current_hidden_nearest_labels(hidden, labels, norms, dim, row, train_count, top_k)
    ids = proposal[:ids]
    target = labels[row]
    top1_hits += 1 if ids[0]? == target
    topk_hits += 1 if ids.includes?(target)
    best_cosines << proposal[:best_cos]
    centroid = current_hidden_centroid_labels(hidden, norms, dim, row, centroid_pack, top_k)
    centroid_ids = centroid[:ids]
    centroid_top1_hits += 1 if centroid_ids[0]? == target
    centroid_topk_hits += 1 if centroid_ids.includes?(target)
    centroid_best_cosines << centroid[:best_cos]
    if ids[0]?
      best_label = ids[0]
      nearest = 0
      best_sim = -Float64::INFINITY
      train_count.times do |j|
        next unless labels[j] == best_label
        sim = hidden_row_cosine(hidden, dim, row, j, norms)
        if sim > best_sim
          best_sim = sim
          nearest = j
        end
      end
      nearest_rows << nearest
    else
      nearest_rows << 0
    end
  end
  proposal_ms = (Time.instant - t_probe).total_milliseconds

  transition_samples = 0
  transition_label_hits = 0
  transition_delta_hits = 0
  t_transition = Time.instant
  (train_count...rows).each_with_index do |row, idx|
    next if row + 1 >= rows
    nearest = nearest_rows[idx]
    next if nearest + 1 >= train_count

    target = labels[row + 1]
    transition_label_hits += 1 if labels[nearest + 1] == target
    pred_hidden = hidden_row_with_delta(hidden, dim, row, nearest, nearest + 1)
    pred_id = ML::GGUF::Qwen35CPU.hidden_top1(weights, pred_hidden)[0]
    transition_delta_hits += 1 if pred_id == target
    transition_samples += 1
  end
  transition_ms = (Time.instant - t_transition).total_milliseconds

  t_pca_transition = Time.instant
  pca_adapter = train_current_hidden_pca_transition(hidden, dim, train_count, pca_transition_rank, pca_iters)
  pca_transition_hits = 0
  pca_transition_samples = 0
  if adapter = pca_adapter
    (train_count...rows).each do |row|
      next if row + 1 >= rows
      target = labels[row + 1]
      inp = hidden_row64(hidden, row, dim)
      pred_delta = predict_block_residual(adapter, inp)
      pred_hidden = Array(Float32).new(dim) { |i| (inp[i] + pred_delta[i]).to_f32 }
      pred_id = ML::GGUF::Qwen35CPU.hidden_top1(weights, pred_hidden)[0]
      pca_transition_hits += 1 if pred_id == target
      pca_transition_samples += 1
    end
  end
  pca_transition_ms = (Time.instant - t_pca_transition).total_milliseconds
  pca_effective_rank = pca_adapter.try(&.input_basis.size) || 0

  sorted_cos = best_cosines.sort
  {
    eval_samples:        eval_count,
    train_samples:       train_count,
    top_k:               top_k,
    top1_hits:           top1_hits,
    topk_hits:           topk_hits,
    centroid_top1_hits:  centroid_top1_hits,
    centroid_topk_hits:  centroid_topk_hits,
    unique_train_labels: labels[0, train_count].uniq.size,
    collect_ms:          collect_ms,
    proposal_ms:         proposal_ms,
    avg_best_cos:        best_cosines.sum / best_cosines.size,
    p50_best_cos:        sorted_cos[sorted_cos.size // 2],
    min_best_cos:        sorted_cos[0],
    centroid_avg_best_cos: centroid_best_cosines.sum / centroid_best_cosines.size,
    top1_rate:           100.0 * top1_hits / eval_count,
    topk_rate:           100.0 * topk_hits / eval_count,
    centroid_top1_rate:  100.0 * centroid_top1_hits / eval_count,
    centroid_topk_rate:  100.0 * centroid_topk_hits / eval_count,
    transition_samples:  transition_samples,
    transition_label_hits: transition_label_hits,
    transition_delta_hits: transition_delta_hits,
    pca_transition_samples: pca_transition_samples,
    pca_transition_hits: pca_transition_hits,
    transition_label_rate: transition_samples > 0 ? 100.0 * transition_label_hits / transition_samples : 0.0,
    transition_delta_rate: transition_samples > 0 ? 100.0 * transition_delta_hits / transition_samples : 0.0,
    pca_transition_rate: pca_transition_samples > 0 ? 100.0 * pca_transition_hits / pca_transition_samples : 0.0,
    transition_ms:       transition_ms,
    pca_transition_ms:   pca_transition_ms,
    pca_transition_effective_rank: pca_effective_rank,
  }
end

private def top1_margin(v : Array(Float32)) : Float64
  best = -Float32::INFINITY
  second = -Float32::INFINITY
  v.each do |x|
    if x > best
      second = best
      best = x
    elsif x > second
      second = x
    end
  end
  (best - second).to_f64
end

private def draft_variant_ffn_top_percent(variant : String) : Int32?
  return nil unless variant.starts_with?("lowrank-ffn-top-")

  percent = variant["lowrank-ffn-top-".size..].to_i? || raise "invalid FFN top-percent variant #{variant.inspect}"
  raise "FFN top-percent must be in 1..100" unless percent >= 1 && percent <= 100
  percent
end

private def draft_variant_ffn_block_top_percent(variant : String) : Int32?
  return nil unless variant.starts_with?("lowrank-ffn-blocktop-")

  percent = variant["lowrank-ffn-blocktop-".size..].to_i? || raise "invalid FFN block-top variant #{variant.inspect}"
  raise "FFN block-top percent must be in 1..100" unless percent >= 1 && percent <= 100
  percent
end

private def draft_variant_ffn_block_pred_percent(variant : String) : Int32?
  return nil unless variant.starts_with?("lowrank-ffn-blockpred-")

  percent = variant["lowrank-ffn-blockpred-".size..].to_i? || raise "invalid FFN block-pred variant #{variant.inspect}"
  raise "FFN block-pred percent must be in 1..100" unless percent >= 1 && percent <= 100
  percent
end

private def draft_variant_ffn_pca_rank(variant : String) : Int32?
  return nil unless variant.starts_with?("lowrank-ffn-pca-")
  return nil if variant.starts_with?("lowrank-ffn-pca-down-")
  return nil if variant.starts_with?("lowrank-ffn-pca-updown-")

  rank = variant["lowrank-ffn-pca-".size..].to_i? || raise "invalid FFN PCA variant #{variant.inspect}"
  raise "FFN PCA rank must be positive" unless rank > 0
  rank
end

private def draft_variant_ffn_pca_down_rank(variant : String) : Int32?
  return nil unless variant.starts_with?("lowrank-ffn-pca-down-")

  rank = variant["lowrank-ffn-pca-down-".size..].to_i? || raise "invalid FFN PCA-down variant #{variant.inspect}"
  raise "FFN PCA-down rank must be positive" unless rank > 0
  rank
end

private def draft_variant_ffn_pca_updown_rank(variant : String) : Int32?
  return nil unless variant.starts_with?("lowrank-ffn-pca-updown-")

  rank = variant["lowrank-ffn-pca-updown-".size..].to_i? || raise "invalid FFN PCA-updown variant #{variant.inspect}"
  raise "FFN PCA-updown rank must be positive" unless rank > 0
  rank
end

private def keep_top_abs_percent!(values : Array(Float32), percent : Int32) : Nil
  return if percent >= 100

  keep = Math.max(1, (values.size.to_i64 * percent + 99) // 100).to_i
  return if keep >= values.size

  threshold = values.map(&.abs).sort![-keep]
  kept = 0
  values.size.times do |i|
    if values[i].abs >= threshold && kept < keep
      kept += 1
    else
      values[i] = 0.0_f32
    end
  end
end

private def keep_top_energy_blocks!(values : Array(Float32), percent : Int32, block_size : Int32) : Nil
  return if percent >= 100
  raise "FFN block size must be positive" unless block_size > 0

  blocks = (values.size + block_size - 1) // block_size
  keep_blocks = Math.max(1, (blocks.to_i64 * percent + 99) // 100).to_i
  return if keep_blocks >= blocks

  energies = Array(Float64).new(blocks, 0.0)
  values.each_with_index do |value, i|
    energies[i // block_size] += value.to_f64 * value.to_f64
  end
  order = (0...blocks).to_a
  order.sort_by! { |block| -energies[block] }
  keep = Set(Int32).new(order[0, keep_blocks])
  blocks.times do |block|
    next if keep.includes?(block)

    start = block * block_size
    stop = Math.min(start + block_size, values.size)
    start.upto(stop - 1) { |i| values[i] = 0.0_f32 }
  end
end

private def zero_except_blocks!(values : Array(Float32), selected : Set(Int32), block_size : Int32) : Nil
  raise "FFN block size must be positive" unless block_size > 0

  blocks = (values.size + block_size - 1) // block_size
  blocks.times do |block|
    next if selected.includes?(block)

    start = block * block_size
    stop = Math.min(start + block_size, values.size)
    start.upto(stop - 1) { |i| values[i] = 0.0_f32 }
  end
end

private def project_vector_with_basis!(values : Array(Float32), basis : Array(Array(Float64)), rank : Int32) : Nil
  return if basis.empty?

  limit = Math.min(rank, basis.size)
  projected = Array(Float64).new(values.size, 0.0)
  limit.times do |i|
    b = basis[i]
    coeff = 0.0
    values.size.times { |d| coeff += values[d].to_f64 * b[d] }
    values.size.times { |d| projected[d] += coeff * b[d] }
  end
  values.size.times { |d| values[d] = projected[d].to_f32 }
end

private def ffn_down_from_adapter(combined : Array(Float32), adapter : FFNAdapter, rank : Int32) : Array(Float32)
  limit = Math.min(rank, adapter.basis.size)
  raise "FFN adapter has no basis vectors" unless limit > 0
  out_dim = adapter.down_basis[0].size
  out = Array(Float32).new(out_dim, 0.0_f32)
  limit.times do |i|
    b = adapter.basis[i]
    coeff = 0.0
    combined.size.times { |d| coeff += combined[d].to_f64 * b[d] }
    coeff_f = coeff.to_f32
    down = adapter.down_basis[i]
    out_dim.times { |d| out[d] += coeff_f * down[d] }
  end
  out
end

private def ffn_out_from_updown_adapter(ffn_in : Array(Float32), adapter : FFNUpDownAdapter, rank : Int32) : Array(Float32)
  adapter.project(ffn_in, rank)
end

private def hadamard_power_of_two?(n : Int32) : Bool
  n > 0 && (n & (n - 1)) == 0
end

private def block_hadamard_inplace!(values : Array(Float64), block_size : Int32) : Nil
  raise "Hadamard block size must be a positive power of two" unless hadamard_power_of_two?(block_size)
  raise "Hadamard vector dimension must be divisible by block size" unless values.size % block_size == 0

  offset = 0
  scale = 1.0 / Math.sqrt(block_size.to_f64)
  while offset < values.size
    width = 1
    while width < block_size
      step = width * 2
      i = 0
      while i < block_size
        width.times do |j|
          a_i = offset + i + j
          b_i = a_i + width
          a = values[a_i]
          b = values[b_i]
          values[a_i] = a + b
          values[b_i] = a - b
        end
        i += step
      end
      width = step
    end
    block_size.times { |i| values[offset + i] *= scale }
    offset += block_size
  end
end

private def quant_dequant_symmetric(values : Array(Float64), bits : Int32) : Array(Float64)
  raise "quant bits must be 2..8" unless bits >= 2 && bits <= 8
  qmax = ((1 << (bits - 1)) - 1).to_f64
  max_abs = values.reduce(0.0) { |m, v| {m, v.abs}.max }
  return Array(Float64).new(values.size, 0.0) if max_abs <= 0.0
  scale = max_abs / qmax
  values.map do |v|
    q = (v / scale).round.clamp(-qmax, qmax)
    q * scale
  end
end

private def quant_dequant_hadamard(values : Array(Float64), bits : Int32, block_size : Int32) : Array(Float64)
  tmp = values.dup
  block_hadamard_inplace!(tmp, block_size)
  tmp = quant_dequant_symmetric(tmp, bits)
  # Normalized Hadamard is self-inverse.
  block_hadamard_inplace!(tmp, block_size)
  tmp
end

private def quantized_updown_adapter(adapter : FFNUpDownAdapter,
                                     bits : Int32,
                                     hadamard_block : Int32? = nil) : FFNUpDownAdapter
  adapter.quantized(bits, hadamard_block)
end

private def relative_rmse(exact : Array(Float32), approx : Array(Float32)) : Float64
  raise "relative_rmse dimension mismatch" unless exact.size == approx.size
  err_sq = 0.0
  exact_sq = 0.0
  exact.size.times do |i|
    e = exact[i].to_f64
    d = approx[i].to_f64 - e
    err_sq += d * d
    exact_sq += e * e
  end
  exact_sq > 0.0 ? Math.sqrt(err_sq / exact_sq) : 0.0
end

private def ffn_updown_hadamard_quant_feature_note(name : String,
                                                   weights : ML::GGUF::Qwen35Weights,
                                                   token_ids : Array(Int32),
                                                   calib_count : Int32,
                                                   layer_ids : Array(Int32),
                                                   adapters : FFNUpDownAdapterMap,
                                                   rank : Int32,
                                                   bits_list : Array(Int32),
                                                   block_sizes : Array(Int32)) : Array(String)
  sample_map = ffn_updown_samples_for_token_sets(weights, [token_ids], layer_ids, token_ids.size)
  notes = [] of String

  layer_ids.uniq.sort.each do |il|
    adapter = adapters[il]? || next
    samples = sample_map[il]? || next
    eval_start = Math.min(calib_count, samples.size)
    next unless eval_start < samples.size
    layer = weights.layers[il]
    next unless layer.is_a?(ML::GGUF::Qwen35RecurrentWeights)

    candidates = [] of NamedTuple(mode: String, adapter: FFNUpDownAdapter)
    bits_list.each do |bits|
      candidates << {mode: "raw_q#{bits}", adapter: quantized_updown_adapter(adapter, bits)}
      block_sizes.each do |block|
        next unless hadamard_power_of_two?(block) && weights.hparams.n_embd % block == 0
        candidates << {mode: "hadamard#{block}_q#{bits}", adapter: quantized_updown_adapter(adapter, bits, block)}
      end
    end

    dense_exact_rel = [] of Float64
    dense_cos = [] of Float64
    rel_by_mode = Hash(String, Array(Float64)).new { |h, k| h[k] = [] of Float64 }
    dense_rel_by_mode = Hash(String, Array(Float64)).new { |h, k| h[k] = [] of Float64 }
    cos_by_mode = Hash(String, Array(Float64)).new { |h, k| h[k] = [] of Float64 }

    samples[eval_start, samples.size - eval_start].each do |sample|
      activation = sample[:activation].map(&.to_f32)
      ffn_in = sample[:ffn_in].map(&.to_f32)
      exact = ML::GGUF::Qwen35CPU.qmatvec_nobias(layer.ffn_down_qw, activation)
      dense = ffn_out_from_updown_adapter(ffn_in, adapter, rank)
      dense_exact_rel << relative_rmse(exact, dense)
      dense_cos << cosine(exact, dense)
      candidates.each do |cand|
        approx = ffn_out_from_updown_adapter(ffn_in, cand[:adapter], rank)
        rel_by_mode[cand[:mode]] << relative_rmse(exact, approx)
        dense_rel_by_mode[cand[:mode]] << relative_rmse(dense, approx)
        cos_by_mode[cand[:mode]] << cosine(exact, approx)
      end
    end

    baseline = dense_exact_rel.empty? ? 0.0 : mean(dense_exact_rel)
    modes = rel_by_mode.keys.sort.map do |mode|
      rel = rel_by_mode[mode]
      dense_rel = dense_rel_by_mode[mode]
      cos = cos_by_mode[mode]
      next "#{mode}:empty" if rel.empty?
      delta = baseline > 0.0 ? ((mean(rel) / baseline) - 1.0) * 100.0 : 0.0
      "#{mode}:rel=#{mean(rel).round(6)},rel_vs_dense=#{mean(dense_rel).round(6)},cos=#{mean(cos).round(6)},delta=#{delta.round(2)}%"
    end
    notes << "ffn_updown_hadamard_quant_features name=#{name} layer=#{il} rank=#{rank} eval_samples=#{dense_exact_rel.size} dense_rel=#{baseline.round(6)} dense_cos=#{mean(dense_cos).round(6)} modes=#{modes.join(' ')}"
  end

  if notes.empty?
    ["ffn_updown_hadamard_quant_features name=#{name} layers=#{layer_ids.join(',')} rank=#{rank} eval_samples=0"]
  else
    notes
  end
end

private def dump_ffn_updown_adapters(path : String,
                                     adapters : FFNUpDownAdapterMap,
                                     rank : Int32,
                                     hidden_dim : Int32,
                                     source : String) : Nil
  ML::GGUF::Qwen35FFNUpDownAdapterArtifact.dump(path, adapters, rank, hidden_dim, source)
end

private struct TopKOracleSample
  getter ids : Array(Int32)
  getter logits : Array(Float32)
  getter exact_id : Int32

  def initialize(@ids : Array(Int32), @logits : Array(Float32), @exact_id : Int32)
  end

  def margin : Float64
    return Float64::INFINITY if @logits.size < 2

    (@logits[0] - @logits[1]).to_f64
  end
end

private def topk_oracle_sample(approx_logits : Array(Float32), exact_id : Int32, top_k : Int32) : TopKOracleSample
  ids = top_k_indices(approx_logits, top_k)
  logits = ids.map { |id| approx_logits[id] }
  TopKOracleSample.new(ids, logits, exact_id)
end

private def softmax_kl(exact : Array(Float32), approx : Array(Float32)) : Float64
  max_exact = exact.max
  max_approx = approx.max
  sum_exact = 0.0
  sum_approx = 0.0
  exact.each { |x| sum_exact += Math.exp((x - max_exact).to_f64) }
  approx.each { |x| sum_approx += Math.exp((x - max_approx).to_f64) }
  log_z_exact = max_exact.to_f64 + Math.log(sum_exact)
  log_z_approx = max_approx.to_f64 + Math.log(sum_approx)
  kl = 0.0
  exact.size.times do |i|
    log_p = exact[i].to_f64 - log_z_exact
    log_q = approx[i].to_f64 - log_z_approx
    p = Math.exp(log_p)
    kl += p * (log_p - log_q)
  end
  kl
end

private def max_abs_delta(a : Array(Float32), b : Array(Float32)) : Float64
  max = 0.0
  a.size.times do |i|
    d = (a[i] - b[i]).to_f64.abs
    max = d if d > max
  end
  max
end

private def simulate_logits(weights : ML::GGUF::Qwen35Weights,
                            token_ids : Array(Int32),
                            target_layer : Int32,
                            bases : Array(Array(Array(Float64))),
                            rank : Int32,
                            calib_count : Int32) : NamedTuple(mean_cos: Float64, min_cos: Float64, max_delta: Float64, top1_match: Float64)
  exact_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: token_ids.size + 2)
  approx_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: token_ids.size + 2)
  lr_state = LowRankState.new
  cosines = [] of Float64
  max_delta = 0.0
  top_matches = 0
  compared = 0

  token_ids.each_with_index do |token_id, pos|
    exact = logits_with_target_layer(weights, token_id, pos.to_i32, exact_state,
      target_layer, bases, rank, calib_count, nil, false)
    approx = logits_with_target_layer(weights, token_id, pos.to_i32, approx_state,
      target_layer, bases, rank, calib_count, lr_state, true)
    next if pos < calib_count

    c = cosine(exact, approx)
    cosines << c
    d = max_abs_delta(exact, approx)
    max_delta = d if d > max_delta
    top_matches += 1 if top1(exact) == top1(approx)
    compared += 1
  end

  {
    mean_cos:   cosines.sum / cosines.size,
    min_cos:    cosines.min,
    max_delta:  max_delta,
    top1_match: 100.0 * top_matches / compared,
  }
end

private def simulate_logits_policy(weights : ML::GGUF::Qwen35Weights,
                                   token_ids : Array(Int32),
                                   layer_bases : LayerBasisMap,
                                   rank : Int32,
                                   calib_count : Int32,
                                   fallback_threshold : Float64?,
                                   refresh_interval : Int32?,
                                   oracle_refresh_interval : Int32?,
                                   output_margin_threshold : Float64?) : NamedTuple(mean_cos: Float64, min_cos: Float64, max_delta: Float64, top1_match: Float64, top5_hit: Float64, mean_kl: Float64, max_kl: Float64, min_margin: Float64, confident_mismatches: Int32, approx_steps: Int32, fallback_steps: Int32, output_fallbacks: Int32)
  exact_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: token_ids.size + 2)
  approx_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: token_ids.size + 2)
  lr_states = {} of Int32 => LowRankState
  cosines = [] of Float64
  kls = [] of Float64
  max_delta = 0.0
  top_matches = 0
  top5_hits = 0
  min_margin = Float64::INFINITY
  confident_mismatches = 0
  output_fallbacks = 0
  compared = 0

  token_ids.each_with_index do |token_id, pos|
    exact = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, false)
    approx = logits_with_lowrank_policy(weights, token_id, pos.to_i32, approx_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, true)
    next if pos < calib_count

    approx_eval = approx
    if threshold = output_margin_threshold
      if top1_margin(approx) < threshold
        output_fallbacks += 1
        approx_eval = exact
      end
    end

    c = cosine(exact, approx_eval)
    cosines << c
    d = max_abs_delta(exact, approx_eval)
    max_delta = d if d > max_delta
    exact_top1 = top1(exact)
    approx_top1 = top1(approx_eval)
    exact_margin = top1_margin(exact)
    min_margin = exact_margin if exact_margin < min_margin
    if exact_top1 == approx_top1
      top_matches += 1
    elsif exact_margin >= 0.5
      confident_mismatches += 1
    end
    top5_hits += 1 if top_k_indices(approx_eval, 5).includes?(exact_top1)
    kls << softmax_kl(exact, approx_eval)
    compared += 1
    if refresh_due?(pos.to_i32, calib_count, oracle_refresh_interval)
      sync_lowrank_shadow!(approx_state, exact_state, layer_bases, lr_states, rank, weights.hparams)
    end
  end

  {
    mean_cos:             cosines.sum / cosines.size,
    min_cos:              cosines.min,
    max_delta:            max_delta,
    top1_match:           100.0 * top_matches / compared,
    top5_hit:             100.0 * top5_hits / compared,
    mean_kl:              kls.sum / kls.size,
    max_kl:               kls.max,
    min_margin:           min_margin,
    confident_mismatches: confident_mismatches,
    approx_steps:         lr_states.values.sum(&.approx_steps),
    fallback_steps:       lr_states.values.sum(&.fallback_steps),
    output_fallbacks:     output_fallbacks,
  }
end

private def simulate_greedy_policy(weights : ML::GGUF::Qwen35Weights,
                                   prompt_ids : Array(Int32),
                                   gen_tokens : Int32,
                                   layer_bases : LayerBasisMap,
                                   rank : Int32,
                                   calib_count : Int32,
                                   fallback_threshold : Float64?,
                                   refresh_interval : Int32?,
                                   oracle_refresh_interval : Int32?,
                                   output_margin_threshold : Float64?) : NamedTuple(mean_cos: Float64, min_cos: Float64, max_delta: Float64, top1_match: Float64, top5_hit: Float64, mean_kl: Float64, max_kl: Float64, min_margin: Float64, confident_mismatches: Int32, approx_steps: Int32, fallback_steps: Int32, output_fallbacks: Int32, exact_ids: Array(Int32), approx_ids: Array(Int32))
  exact_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: prompt_ids.size + gen_tokens + 2)
  approx_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: prompt_ids.size + gen_tokens + 2)
  lr_states = {} of Int32 => LowRankState
  exact_logits = [] of Float32
  approx_logits = [] of Float32

  prompt_ids.each_with_index do |token_id, pos|
    exact_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, false)
    approx_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, approx_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, true)
    if refresh_due?(pos.to_i32, calib_count, oracle_refresh_interval)
      sync_lowrank_shadow!(approx_state, exact_state, layer_bases, lr_states, rank, weights.hparams)
      approx_logits = exact_logits.dup
    end
  end

  cosines = [] of Float64
  kls = [] of Float64
  max_delta = 0.0
  top_matches = 0
  top5_hits = 0
  min_margin = Float64::INFINITY
  confident_mismatches = 0
  output_fallbacks = 0
  exact_ids = [] of Int32
  approx_ids = [] of Int32

  gen_tokens.times do |step|
    exact_top1 = top1(exact_logits)
    approx_eval = approx_logits
    if threshold = output_margin_threshold
      if top1_margin(approx_logits) < threshold
        output_fallbacks += 1
        approx_eval = exact_logits
      end
    end
    approx_top1 = top1(approx_eval)
    exact_ids << exact_top1
    approx_ids << approx_top1

    c = cosine(exact_logits, approx_eval)
    cosines << c
    d = max_abs_delta(exact_logits, approx_eval)
    max_delta = d if d > max_delta
    exact_margin = top1_margin(exact_logits)
    min_margin = exact_margin if exact_margin < min_margin
    if exact_top1 == approx_top1
      top_matches += 1
    elsif exact_margin >= 0.5
      confident_mismatches += 1
    end
    top5_hits += 1 if top_k_indices(approx_eval, 5).includes?(exact_top1)
    kls << softmax_kl(exact_logits, approx_eval)

    pos = prompt_ids.size + step
    # Teacher-forced on the exact greedy trajectory. This isolates policy drift
    # from cascading different-token hidden-state divergence.
    exact_logits = logits_with_lowrank_policy(weights, exact_top1, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, false)
    approx_logits = logits_with_lowrank_policy(weights, exact_top1, pos.to_i32, approx_state,
      layer_bases, rank, calib_count, lr_states, fallback_threshold, refresh_interval, true)
    if refresh_due?(pos.to_i32, calib_count, oracle_refresh_interval)
      sync_lowrank_shadow!(approx_state, exact_state, layer_bases, lr_states, rank, weights.hparams)
      approx_logits = exact_logits.dup
    end
  end

  {
    mean_cos:             cosines.sum / cosines.size,
    min_cos:              cosines.min,
    max_delta:            max_delta,
    top1_match:           100.0 * top_matches / gen_tokens,
    top5_hit:             100.0 * top5_hits / gen_tokens,
    mean_kl:              kls.sum / kls.size,
    max_kl:               kls.max,
    min_margin:           min_margin,
    confident_mismatches: confident_mismatches,
    approx_steps:         lr_states.values.sum(&.approx_steps),
    fallback_steps:       lr_states.values.sum(&.fallback_steps),
    output_fallbacks:     output_fallbacks,
    exact_ids:            exact_ids,
    approx_ids:           approx_ids,
  }
end

private def simulate_block_surrogate_logits_policy(weights : ML::GGUF::Qwen35Weights,
                                                   token_ids : Array(Int32),
                                                   block_start : Int32,
                                                   block_end : Int32,
                                                   adapter : BlockResidualSurrogate | BlockResidualMixture,
                                                   calib_count : Int32,
                                                   state_mode : String) : NamedTuple(mean_cos: Float64, min_cos: Float64, max_delta: Float64, top1_match: Float64, top5_hit: Float64, mean_kl: Float64, max_kl: Float64, min_margin: Float64, confident_mismatches: Int32, approx_blocks: Int32, skipped_layers: Int32)
  exact_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: token_ids.size + 2)
  approx_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: token_ids.size + 2)
  cosines = [] of Float64
  kls = [] of Float64
  max_delta = 0.0
  top_matches = 0
  top5_hits = 0
  min_margin = Float64::INFINITY
  confident_mismatches = 0
  compared = 0

  token_ids.each_with_index do |token_id, pos|
    exact = logits_with_block_surrogate_policy(weights, token_id, pos.to_i32, exact_state, block_start, block_end, adapter, calib_count, false, state_mode)
    approx = logits_with_block_surrogate_policy(weights, token_id, pos.to_i32, approx_state, block_start, block_end, adapter, calib_count, true, state_mode)
    next if pos < calib_count

    cmp = compare_logit_rows(exact, approx, cosines, kls)
    max_delta = cmp[:max_delta] if cmp[:max_delta] > max_delta
    top_matches += 1 if cmp[:top1_match]
    top5_hits += 1 if cmp[:top5_hit]
    min_margin = cmp[:margin] if cmp[:margin] < min_margin
    confident_mismatches += 1 if cmp[:confident_mismatch]
    compared += 1
  end

  {
    mean_cos:             cosines.sum / cosines.size,
    min_cos:              cosines.min,
    max_delta:            max_delta,
    top1_match:           100.0 * top_matches / compared,
    top5_hit:             100.0 * top5_hits / compared,
    mean_kl:              kls.sum / kls.size,
    max_kl:               kls.max,
    min_margin:           min_margin,
    confident_mismatches: confident_mismatches,
    approx_blocks:        compared,
    skipped_layers:       compared * (block_end - block_start + 1),
  }
end

private def simulate_block_surrogate_greedy_policy(weights : ML::GGUF::Qwen35Weights,
                                                   prompt_ids : Array(Int32),
                                                   gen_tokens : Int32,
                                                   block_start : Int32,
                                                   block_end : Int32,
                                                   adapter : BlockResidualSurrogate | BlockResidualMixture,
                                                   calib_count : Int32,
                                                   state_mode : String) : NamedTuple(mean_cos: Float64, min_cos: Float64, max_delta: Float64, top1_match: Float64, top5_hit: Float64, mean_kl: Float64, max_kl: Float64, min_margin: Float64, confident_mismatches: Int32, approx_blocks: Int32, skipped_layers: Int32, exact_ids: Array(Int32), approx_ids: Array(Int32))
  exact_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: prompt_ids.size + gen_tokens + 2)
  approx_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: prompt_ids.size + gen_tokens + 2)
  exact_logits = [] of Float32
  approx_logits = [] of Float32

  prompt_ids.each_with_index do |token_id, pos|
    exact_logits = logits_with_block_surrogate_policy(weights, token_id, pos.to_i32, exact_state, block_start, block_end, adapter, calib_count, false, state_mode)
    approx_logits = logits_with_block_surrogate_policy(weights, token_id, pos.to_i32, approx_state, block_start, block_end, adapter, calib_count, true, state_mode)
  end

  cosines = [] of Float64
  kls = [] of Float64
  max_delta = 0.0
  top_matches = 0
  top5_hits = 0
  min_margin = Float64::INFINITY
  confident_mismatches = 0
  exact_ids = [] of Int32
  approx_ids = [] of Int32

  gen_tokens.times do |step|
    exact_top1 = top1(exact_logits)
    approx_top1 = top1(approx_logits)
    exact_ids << exact_top1
    approx_ids << approx_top1

    cmp = compare_logit_rows(exact_logits, approx_logits, cosines, kls)
    max_delta = cmp[:max_delta] if cmp[:max_delta] > max_delta
    top_matches += 1 if cmp[:top1_match]
    top5_hits += 1 if cmp[:top5_hit]
    min_margin = cmp[:margin] if cmp[:margin] < min_margin
    confident_mismatches += 1 if cmp[:confident_mismatch]

    pos = prompt_ids.size + step
    exact_logits = logits_with_block_surrogate_policy(weights, exact_top1, pos.to_i32, exact_state, block_start, block_end, adapter, calib_count, false, state_mode)
    # Teacher-forced on the exact greedy token to isolate hidden/state drift
    # from different-token cascade.
    approx_logits = logits_with_block_surrogate_policy(weights, exact_top1, pos.to_i32, approx_state, block_start, block_end, adapter, calib_count, true, state_mode)
  end

  approx_blocks = gen_tokens + prompt_ids.size - calib_count
  {
    mean_cos:             cosines.sum / cosines.size,
    min_cos:              cosines.min,
    max_delta:            max_delta,
    top1_match:           100.0 * top_matches / gen_tokens,
    top5_hit:             100.0 * top5_hits / gen_tokens,
    mean_kl:              kls.sum / kls.size,
    max_kl:               kls.max,
    min_margin:           min_margin,
    confident_mismatches: confident_mismatches,
    approx_blocks:        approx_blocks,
    skipped_layers:       approx_blocks * (block_end - block_start + 1),
    exact_ids:            exact_ids,
    approx_ids:           approx_ids,
  }
end

private def exact_greedy_ids_with_block_policy(weights : ML::GGUF::Qwen35Weights,
                                               prompt_ids : Array(Int32),
                                               gen_tokens : Int32,
                                               block_start : Int32,
                                               block_end : Int32,
                                               adapter : BlockResidualSurrogate | BlockResidualMixture,
                                               calib_count : Int32,
                                               state_mode : String,
                                               max_seq : Int32) : Array(Int32)
  state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
  logits = [] of Float32
  prompt_ids.each_with_index do |token_id, pos|
    logits = logits_with_block_surrogate_policy(weights, token_id, pos.to_i32, state,
      block_start, block_end, adapter, calib_count, false, state_mode)
  end

  ids = [] of Int32
  gen_tokens.times do |step|
    id = top1(logits)
    ids << id
    if step + 1 < gen_tokens
      pos = prompt_ids.size + step
      logits = logits_with_block_surrogate_policy(weights, id, pos.to_i32, state,
        block_start, block_end, adapter, calib_count, false, state_mode)
    end
  end
  ids
end

private def exact_greedy_decode_with_block_policy_timed(weights : ML::GGUF::Qwen35Weights,
                                                        prompt_ids : Array(Int32),
                                                        gen_tokens : Int32,
                                                        block_start : Int32,
                                                        block_end : Int32,
                                                        adapter : BlockResidualSurrogate | BlockResidualMixture,
                                                        calib_count : Int32,
                                                        state_mode : String,
                                                        max_seq : Int32)
  state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
  logits = [] of Float32
  t_prompt = Time.instant
  prompt_ids.each_with_index do |token_id, pos|
    logits = logits_with_block_surrogate_policy(weights, token_id, pos.to_i32, state,
      block_start, block_end, adapter, calib_count, false, state_mode)
  end
  prompt_ms = (Time.instant - t_prompt).total_milliseconds

  ids = [] of Int32
  t_decode = Time.instant
  gen_tokens.times do |step|
    id = top1(logits)
    ids << id
    if step + 1 < gen_tokens
      pos = prompt_ids.size + step
      logits = logits_with_block_surrogate_policy(weights, id, pos.to_i32, state,
        block_start, block_end, adapter, calib_count, false, state_mode)
    end
  end
  decode_ms = (Time.instant - t_decode).total_milliseconds

  {ids: ids, prompt_ms: prompt_ms, decode_ms: decode_ms}
end

private def simulate_block_surrogate_self_spec_policy(weights : ML::GGUF::Qwen35Weights,
                                                      prompt_ids : Array(Int32),
                                                      gen_tokens : Int32,
                                                      gamma : Int32,
                                                      block_start : Int32,
                                                      block_end : Int32,
                                                      adapter : BlockResidualSurrogate | BlockResidualMixture,
                                                      calib_count : Int32,
                                                      state_mode : String)
  raise "block surrogate self-spec gamma must be positive" unless gamma > 0
  raise "block surrogate self-spec needs positive gen_tokens" unless gen_tokens > 0
  raise "block surrogate self-spec needs a non-empty prompt" if prompt_ids.empty?

  hp = weights.hparams
  max_seq = prompt_ids.size + gen_tokens + gamma + 4
  verifier_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  exact_logits = [] of Float32
  state_before_last = verifier_state.fork
  last_token = prompt_ids[0]
  pos_last = 0

  prompt_ms = 0.0
  draft_fork_ms = 0.0
  draft_ms = 0.0
  verifier_fork_ms = 0.0
  verifier_ms = 0.0

  t_prompt = Time.instant
  prompt_ids.each_with_index do |token_id, pos|
    state_before_last = verifier_state.fork
    last_token = token_id
    pos_last = pos.to_i32
    exact_logits = logits_with_block_surrogate_policy(weights, token_id, pos.to_i32, verifier_state,
      block_start, block_end, adapter, calib_count, false, state_mode)
  end
  prompt_ms = (Time.instant - t_prompt).total_milliseconds

  chunks = 0
  full_accept_chunks = 0
  rejections = 0
  proposed_tokens = 0
  accepted_draft_tokens = 0
  verifier_tokens = 0
  correction_steps = 0
  draft_top2_hits = 0
  draft_top5_hits = 0
  gamma_history = [] of Int32
  accept_history = [] of Int32
  draft_min_margin_history = [] of Float64
  exact_ids = [] of Int32
  emitted_ids = [] of Int32
  draft_ids = [] of Int32

  while emitted_ids.size < gen_tokens
    chunks += 1
    proposal_limit = Math.min(gamma, gen_tokens - emitted_ids.size)
    gamma_history << proposal_limit

    t_draft_fork = Time.instant
    draft_state = state_before_last.fork
    draft_fork_ms += (Time.instant - t_draft_fork).total_milliseconds
    t_draft = Time.instant
    draft_logits = logits_with_block_surrogate_policy(weights, last_token, pos_last, draft_state,
      block_start, block_end, adapter, calib_count, true, state_mode)
    proposal = [] of Int32
    proposal_top2 = [] of Array(Int32)
    proposal_top5 = [] of Array(Int32)
    min_draft_margin = Float64::INFINITY
    proposal_limit.times do |i|
      candidate = top1(draft_logits)
      proposal << candidate
      draft_ids << candidate
      proposed_tokens += 1
      proposal_top2 << top_k_indices(draft_logits, 2)
      proposal_top5 << top_k_indices(draft_logits, 5)
      margin = top1_margin(draft_logits)
      min_draft_margin = margin if margin < min_draft_margin
      if i + 1 < proposal_limit
        draft_pos = prompt_ids.size + emitted_ids.size + i
        draft_logits = logits_with_block_surrogate_policy(weights, candidate, draft_pos.to_i32, draft_state,
          block_start, block_end, adapter, calib_count, true, state_mode)
      end
    end
    draft_ms += (Time.instant - t_draft).total_milliseconds
    draft_min_margin_history << min_draft_margin

    rejected = false
    accepted_this = 0
    proposal.each_with_index do |candidate, i|
      t_verify_top = Time.instant
      expected = top1(exact_logits)
      verifier_ms += (Time.instant - t_verify_top).total_milliseconds
      exact_ids << expected
      draft_top2_hits += 1 if proposal_top2[i].includes?(expected)
      draft_top5_hits += 1 if proposal_top5[i].includes?(expected)

      verify_pos = prompt_ids.size + emitted_ids.size
      emitted_ids << expected
      verifier_tokens += 1
      if candidate == expected
        accepted_draft_tokens += 1
        accepted_this += 1
      else
        rejections += 1
        correction_steps += 1
        rejected = true
      end

      last_token = expected
      pos_last = verify_pos.to_i32
      if emitted_ids.size < gen_tokens
        t_verify_fork = Time.instant
        state_before_last = verifier_state.fork
        verifier_fork_ms += (Time.instant - t_verify_fork).total_milliseconds
        t_verify = Time.instant
        exact_logits = logits_with_block_surrogate_policy(weights, expected, verify_pos.to_i32, verifier_state,
          block_start, block_end, adapter, calib_count, false, state_mode)
        verifier_ms += (Time.instant - t_verify).total_milliseconds
      end

      break if rejected || emitted_ids.size >= gen_tokens
    end

    full_accept_chunks += 1 unless rejected
    accept_history << accepted_this
  end

  baseline = exact_greedy_decode_with_block_policy_timed(weights, prompt_ids, gen_tokens,
    block_start, block_end, adapter, calib_count, state_mode, max_seq)
  baseline_ids = baseline[:ids]
  self_seq_decode_ms = draft_fork_ms + draft_ms + verifier_fork_ms + verifier_ms
  ideal_overlap_decode_ms = Math.max(draft_fork_ms + draft_ms, verifier_fork_ms + verifier_ms)
  {
    chunks:                   chunks,
    full_accept_chunks:       full_accept_chunks,
    rejections:               rejections,
    emitted_tokens:           emitted_ids.size,
    proposed_tokens:          proposed_tokens,
    accepted_draft_tokens:    accepted_draft_tokens,
    verifier_tokens:          verifier_tokens,
    correction_steps:         correction_steps,
    accept_rate:              proposed_tokens > 0 ? 100.0 * accepted_draft_tokens / proposed_tokens : 0.0,
    avg_accept:               chunks > 0 ? accepted_draft_tokens.to_f64 / chunks : 0.0,
    draft_top2_hit_rate:      verifier_tokens > 0 ? 100.0 * draft_top2_hits / verifier_tokens : 0.0,
    draft_top5_hit_rate:      verifier_tokens > 0 ? 100.0 * draft_top5_hits / verifier_tokens : 0.0,
    gamma_history:            gamma_history,
    accept_history:           accept_history,
    draft_min_margin_history: draft_min_margin_history,
    prompt_ms:                prompt_ms,
    baseline_prompt_ms:       baseline[:prompt_ms],
    baseline_decode_ms:       baseline[:decode_ms],
    draft_ms:                 draft_ms,
    draft_fork_ms:            draft_fork_ms,
    verifier_ms:              verifier_ms,
    verifier_fork_ms:         verifier_fork_ms,
    self_seq_decode_ms:       self_seq_decode_ms,
    ideal_overlap_decode_ms:  ideal_overlap_decode_ms,
    cpu_seq_speedup:          self_seq_decode_ms > 0.0 ? baseline[:decode_ms] / self_seq_decode_ms : 0.0,
    ideal_overlap_speedup:    ideal_overlap_decode_ms > 0.0 ? baseline[:decode_ms] / ideal_overlap_decode_ms : 0.0,
    parity:                   emitted_ids == baseline_ids,
    verifier_parity:          exact_ids == baseline_ids,
    exact_ids:                exact_ids,
    emitted_ids:              emitted_ids,
    baseline_ids:             baseline_ids,
    draft_ids:                draft_ids,
  }
end

private def simulate_block_surrogate_tree_oracle(weights : ML::GGUF::Qwen35Weights,
                                                 prompt_ids : Array(Int32),
                                                 gen_tokens : Int32,
                                                 top_k : Int32,
                                                 progressive_schedule : Array(Int32),
                                                 block_start : Int32,
                                                 block_end : Int32,
                                                 adapter : BlockResidualSurrogate | BlockResidualMixture,
                                                 calib_count : Int32,
                                                 state_mode : String,
                                                 warmup_tokens : Int32 = 0,
                                                 prefill_seed : Bool = false,
                                                 branch_verify : Bool = false,
                                                 select_advance : Bool = false) : NamedTuple(chunks: Int32, full_rescue_chunks: Int32, misses: Int32, emitted_tokens: Int32, warmup_tokens: Int32, prefill_seed: Bool, branch_verify: Bool, select_advance: Bool, prefill_seed_tokens: Int32, tree_tokens: Int32, draft_steps: Int32, top1_hits: Int32, topk_hits: Int32, branch_tokens_rank: Int32, branch_tokens_full: Int32, branch_verify_attempts: Int32, branch_verify_wasted_attempts: Int32, branch_verify_corrections: Int32, branch_verify_ms: Float64, branch_verify_fork_ms: Float64, branch_verify_forward_ms: Float64, correction_steps: Int32, top1_rate: Float64, topk_rate: Float64, avg_rank_branch_tokens: Float64, avg_full_branch_tokens: Float64, avg_rank_branch_tokens_total: Float64, avg_full_branch_tokens_total: Float64, schedule_history: Array(Int32), exact_ids: Array(Int32), emitted_ids: Array(Int32))
  raise "block surrogate tree top_k must be >= 2" unless top_k >= 2
  raise "block surrogate tree top_k must be <= 16" unless top_k <= 16
  raise "block surrogate tree schedule must not be empty" if progressive_schedule.empty?
  raise "block surrogate tree schedule values must be positive" if progressive_schedule.any? { |v| v <= 0 }
  raise "block surrogate tree needs positive gen_tokens" unless gen_tokens > 0
  raise "block surrogate tree needs a non-empty prompt" if prompt_ids.empty?
  raise "block surrogate tree warmup must be non-negative" unless warmup_tokens >= 0
  raise "block surrogate tree branch modes are mutually exclusive" if branch_verify && select_advance
  effective_warmup_tokens = prefill_seed ? Math.max(warmup_tokens, 1) : warmup_tokens

  hp = weights.hparams
  max_gamma = progressive_schedule.max
  max_seq = prompt_ids.size + gen_tokens + max_gamma + 4
  exact_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  exact_logits = [] of Float32
  state_before_last = exact_state.fork
  last_token = prompt_ids[0]
  pos_last = 0

  prompt_ids.each_with_index do |token_id, pos|
    state_before_last = exact_state.fork
    last_token = token_id
    pos_last = pos.to_i32
    exact_logits = logits_with_block_surrogate_policy(weights, token_id, pos.to_i32, exact_state,
      block_start, block_end, adapter, calib_count, false, state_mode)
  end

  chunks = 0
  full_rescue_chunks = 0
  misses = 0
  emitted_tokens = 0
  draft_steps = 0
  top1_hits = 0
  topk_hits = 0
  branch_tokens_rank = 0
  branch_tokens_full = 0
  branch_verify_attempts = 0
  branch_verify_wasted_attempts = 0
  branch_verify_corrections = 0
  branch_verify_ms = 0.0
  branch_verify_fork_ms = 0.0
  branch_verify_forward_ms = 0.0
  correction_steps = 0
  progressive_index = 0
  schedule_history = [] of Int32
  exact_ids = [] of Int32
  emitted_ids = [] of Int32

  while emitted_tokens < gen_tokens
    if emitted_tokens < effective_warmup_tokens
      exact_top1 = top1(exact_logits)
      exact_ids << exact_top1
      emitted_tokens += 1
      emitted_ids << exact_top1
      state_before_last = exact_state.fork
      last_token = exact_top1
      pos_last = (prompt_ids.size + emitted_tokens - 1).to_i32
      if emitted_tokens < gen_tokens
        exact_logits = logits_with_block_surrogate_policy(weights, exact_top1, pos_last, exact_state,
          block_start, block_end, adapter, calib_count, false, state_mode)
      end
      next
    end

    chunks += 1
    chunk_gamma = Math.min(progressive_schedule[progressive_index], gen_tokens - emitted_tokens)
    schedule_history << chunk_gamma
    draft_state = state_before_last.fork
    draft_logits = logits_with_block_surrogate_policy(weights, last_token, pos_last, draft_state,
      block_start, block_end, adapter, calib_count, true, state_mode)

    rescued_chunk = true
    chunk_gamma.times do |j|
      exact_top1 = top1(exact_logits)
      exact_ids << exact_top1
      draft_topk = top_k_indices(draft_logits, top_k)
      current_pos = (prompt_ids.size + emitted_tokens).to_i32
      draft_steps += 1
      if draft_topk[0]? == exact_top1
        top1_hits += 1
        topk_hits += 1
        branch_tokens_rank += 1
        branch_tokens_full += 1
      elsif idx = draft_topk.index(exact_top1)
        topk_hits += 1
        branch_tokens_rank += idx + 1
        branch_tokens_full += top_k
      else
        misses += 1
        correction_steps += 1
        branch_tokens_rank += top_k
        branch_tokens_full += top_k
        rescued_chunk = false
      end

      emitted_tokens += 1
      emitted_ids << exact_top1
      last_token = exact_top1
      pos_last = current_pos
      if emitted_tokens < gen_tokens
        if select_advance
          t_branch = Time.instant
          t_base = Time.instant
          branch_base_state = exact_state.fork
          branch_verify_fork_ms += (Time.instant - t_base).total_milliseconds
          t_fork = Time.instant
          branch_state = branch_base_state.fork
          branch_verify_fork_ms += (Time.instant - t_fork).total_milliseconds
          t_forward = Time.instant
          exact_logits = logits_with_block_surrogate_policy(weights, exact_top1, current_pos, branch_state,
            block_start, block_end, adapter, calib_count, false, state_mode)
          branch_verify_forward_ms += (Time.instant - t_forward).total_milliseconds
          branch_verify_attempts += 1
          branch_verify_corrections += 1 unless draft_topk.includes?(exact_top1)
          exact_state = branch_state
          state_before_last = branch_base_state
          branch_verify_ms += (Time.instant - t_branch).total_milliseconds
        elsif branch_verify
          t_branch = Time.instant
          t_base = Time.instant
          branch_base_state = exact_state.fork
          branch_verify_fork_ms += (Time.instant - t_base).total_milliseconds
          selected = false
          draft_topk.each do |candidate|
            t_fork = Time.instant
            branch_state = branch_base_state.fork
            branch_verify_fork_ms += (Time.instant - t_fork).total_milliseconds
            t_forward = Time.instant
            candidate_logits = logits_with_block_surrogate_policy(weights, candidate, current_pos, branch_state,
              block_start, block_end, adapter, calib_count, false, state_mode)
            branch_verify_forward_ms += (Time.instant - t_forward).total_milliseconds
            branch_verify_attempts += 1
            if candidate == exact_top1
              exact_state = branch_state
              exact_logits = candidate_logits
              selected = true
              break
            end
            branch_verify_wasted_attempts += 1
          end
          unless selected
            t_fork = Time.instant
            branch_state = branch_base_state.fork
            branch_verify_fork_ms += (Time.instant - t_fork).total_milliseconds
            t_forward = Time.instant
            exact_logits = logits_with_block_surrogate_policy(weights, exact_top1, current_pos, branch_state,
              block_start, block_end, adapter, calib_count, false, state_mode)
            branch_verify_forward_ms += (Time.instant - t_forward).total_milliseconds
            branch_verify_attempts += 1
            branch_verify_corrections += 1
            exact_state = branch_state
          end
          state_before_last = branch_base_state
          branch_verify_ms += (Time.instant - t_branch).total_milliseconds
        else
          state_before_last = exact_state.fork
          exact_logits = logits_with_block_surrogate_policy(weights, exact_top1, pos_last, exact_state,
            block_start, block_end, adapter, calib_count, false, state_mode)
        end
      end
      break if !rescued_chunk || emitted_tokens >= gen_tokens || j == chunk_gamma - 1

      draft_logits = logits_with_block_surrogate_policy(weights, exact_top1, pos_last, draft_state,
        block_start, block_end, adapter, calib_count, true, state_mode)
    end

    full_rescue_chunks += 1 if rescued_chunk
    progressive_index = rescued_chunk ? ((progressive_index + 1) % progressive_schedule.size) : 0
  end

  {
    chunks:                 chunks,
    full_rescue_chunks:     full_rescue_chunks,
    misses:                 misses,
    emitted_tokens:         emitted_tokens,
    warmup_tokens:          Math.min(effective_warmup_tokens, gen_tokens),
    prefill_seed:           prefill_seed,
    branch_verify:          branch_verify,
    select_advance:         select_advance,
    prefill_seed_tokens:    prefill_seed ? Math.min(1, gen_tokens) : 0,
    tree_tokens:            draft_steps,
    draft_steps:            draft_steps,
    top1_hits:              top1_hits,
    topk_hits:              topk_hits,
    branch_tokens_rank:     branch_tokens_rank,
    branch_tokens_full:     branch_tokens_full,
    branch_verify_attempts: branch_verify_attempts,
    branch_verify_wasted_attempts: branch_verify_wasted_attempts,
    branch_verify_corrections: branch_verify_corrections,
    branch_verify_ms:       branch_verify_ms,
    branch_verify_fork_ms:  branch_verify_fork_ms,
    branch_verify_forward_ms: branch_verify_forward_ms,
    correction_steps:       correction_steps,
    top1_rate:              draft_steps > 0 ? 100.0 * top1_hits / draft_steps : 0.0,
    topk_rate:              draft_steps > 0 ? 100.0 * topk_hits / draft_steps : 0.0,
    avg_rank_branch_tokens: draft_steps > 0 ? branch_tokens_rank.to_f64 / draft_steps : 0.0,
    avg_full_branch_tokens: draft_steps > 0 ? branch_tokens_full.to_f64 / draft_steps : 0.0,
    avg_rank_branch_tokens_total: emitted_tokens > 0 ? branch_tokens_rank.to_f64 / emitted_tokens : 0.0,
    avg_full_branch_tokens_total: emitted_tokens > 0 ? branch_tokens_full.to_f64 / emitted_tokens : 0.0,
    schedule_history:       schedule_history,
    exact_ids:              exact_ids,
    emitted_ids:            emitted_ids,
  }
end

private def block_surrogate_tree_metrics_summary(tree) : String
  "prefill_seed=#{tree[:prefill_seed]} branch_verify=#{tree[:branch_verify]} " \
    "select_advance=#{tree[:select_advance]} " \
    "prefill_seed_tokens=#{tree[:prefill_seed_tokens]} " \
    "warmup_tokens=#{tree[:warmup_tokens]} tree_tokens=#{tree[:tree_tokens]} " \
    "chunks=#{tree[:chunks]} full_rescue_chunks=#{tree[:full_rescue_chunks]} misses=#{tree[:misses]} " \
    "draft_steps=#{tree[:draft_steps]} top1_hits=#{tree[:top1_hits]} topk_hits=#{tree[:topk_hits]} " \
    "top1_rate=#{tree[:top1_rate].round(2)}% topk_rate=#{tree[:topk_rate].round(2)}% " \
    "branch_tokens_rank=#{tree[:branch_tokens_rank]} branch_tokens_full=#{tree[:branch_tokens_full]} " \
    "avg_rank_branch_tokens=#{tree[:avg_rank_branch_tokens].round(3)} " \
    "avg_full_branch_tokens=#{tree[:avg_full_branch_tokens].round(3)} " \
    "avg_rank_branch_tokens_total=#{tree[:avg_rank_branch_tokens_total].round(3)} " \
    "avg_full_branch_tokens_total=#{tree[:avg_full_branch_tokens_total].round(3)} " \
    "branch_verify_attempts=#{tree[:branch_verify_attempts]} " \
    "branch_verify_wasted_attempts=#{tree[:branch_verify_wasted_attempts]} " \
    "branch_verify_corrections=#{tree[:branch_verify_corrections]} " \
    "branch_verify_ms=#{tree[:branch_verify_ms].round(3)} " \
    "branch_verify_fork_ms=#{tree[:branch_verify_fork_ms].round(3)} " \
    "branch_verify_forward_ms=#{tree[:branch_verify_forward_ms].round(3)} " \
    "correction_steps=#{tree[:correction_steps]} schedule_history=#{tree[:schedule_history].join(',')} " \
    "exact_ids=#{tree[:exact_ids].join(',')} emitted_ids=#{tree[:emitted_ids].join(',')}"
end

private def simulate_block_surrogate_topk_oracle_calibration(weights : ML::GGUF::Qwen35Weights,
                                                            prompt_ids : Array(Int32),
                                                            gen_tokens : Int32,
                                                            top_k : Int32,
                                                            train_tokens : Int32?,
                                                            block_start : Int32,
                                                            block_end : Int32,
                                                            adapter : BlockResidualSurrogate | BlockResidualMixture,
                                                            calib_count : Int32,
                                                            state_mode : String) : NamedTuple(samples: Int32, train_samples: Int32, test_samples: Int32, best_token_scale: Float64, best_rank_scale: Float64, best_margin_threshold: Float64, train_top1_rate: Float64, train_topk_rate: Float64, train_avg_branch_tokens: Float64, baseline_top1_rate: Float64, baseline_topk_rate: Float64, baseline_avg_branch_tokens: Float64, calibrated_top1_rate: Float64, calibrated_topk_rate: Float64, calibrated_avg_branch_tokens: Float64, margin_gate_rate: Float64, margin_gate_topk_rate: Float64, margin_gate_avg_branch_tokens: Float64, margin_gate_misses: Int32, margin_gate_cost: Float64, baseline_misses: Int32, calibrated_misses: Int32, exact_ids: Array(Int32))
  raise "block surrogate topK oracle top_k must be >= 2" unless top_k >= 2
  raise "block surrogate topK oracle top_k must be <= 16" unless top_k <= 16
  raise "block surrogate topK oracle gen_tokens must be >= 4" unless gen_tokens >= 4
  raise "block surrogate topK oracle needs a non-empty prompt" if prompt_ids.empty?

  hp = weights.hparams
  max_seq = prompt_ids.size + gen_tokens + 4
  exact_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  exact_logits = [] of Float32
  state_before_last = exact_state.fork
  last_token = prompt_ids[0]
  pos_last = 0

  prompt_ids.each_with_index do |token_id, pos|
    state_before_last = exact_state.fork
    last_token = token_id
    pos_last = pos.to_i32
    exact_logits = logits_with_block_surrogate_policy(weights, token_id, pos.to_i32, exact_state,
      block_start, block_end, adapter, calib_count, false, state_mode)
  end

  draft_state = state_before_last.fork
  draft_logits = logits_with_block_surrogate_policy(weights, last_token, pos_last, draft_state,
    block_start, block_end, adapter, calib_count, true, state_mode)

  samples = [] of TopKOracleSample
  exact_ids = [] of Int32
  gen_tokens.times do |step|
    exact_top1 = top1(exact_logits)
    samples << topk_oracle_sample(draft_logits, exact_top1, top_k)
    exact_ids << exact_top1

    pos_last = (prompt_ids.size + step).to_i32
    break if step == gen_tokens - 1

    exact_logits = logits_with_block_surrogate_policy(weights, exact_top1, pos_last, exact_state,
      block_start, block_end, adapter, calib_count, false, state_mode)
    draft_logits = logits_with_block_surrogate_policy(weights, exact_top1, pos_last, draft_state,
      block_start, block_end, adapter, calib_count, true, state_mode)
  end

  requested_train = train_tokens || (samples.size // 2)
  train_count = requested_train.clamp(1, samples.size - 1)
  train = samples[0, train_count]
  test = samples[train_count, samples.size - train_count]
  biases = train_topk_oracle_biases(train, top_k)
  zero_token_bias = {} of Int32 => Float64
  zero_rank_bias = Array(Float64).new(top_k, 0.0)
  baseline = eval_topk_oracle_samples(test, zero_token_bias, zero_rank_bias, 0.0, 0.0)

  token_scales = [0.0, 0.25, 0.5, 1.0, 2.0]
  rank_scales = [0.0, 0.25, 0.5, 1.0]
  best_token_scale = 0.0
  best_rank_scale = 0.0
  best_train = eval_topk_oracle_samples(train, biases[:token_bias], biases[:rank_bias], 0.0, 0.0)
  token_scales.each do |ts|
    rank_scales.each do |rs|
      cur = eval_topk_oracle_samples(train, biases[:token_bias], biases[:rank_bias], ts, rs)
      if cur[:avg_branch_tokens] < best_train[:avg_branch_tokens] ||
         (cur[:avg_branch_tokens] == best_train[:avg_branch_tokens] && cur[:top1_rate] > best_train[:top1_rate])
        best_train = cur
        best_token_scale = ts
        best_rank_scale = rs
      end
    end
  end

  calibrated = eval_topk_oracle_samples(test, biases[:token_bias], biases[:rank_bias], best_token_scale, best_rank_scale)
  correction_penalty = top_k.to_f64
  thresholds = [-1.0] + train.map(&.margin).uniq.sort + [Float64::INFINITY]
  best_margin_threshold = thresholds[0]
  best_margin_train = eval_topk_margin_gate(train, best_margin_threshold, correction_penalty)
  thresholds.each do |threshold|
    cur = eval_topk_margin_gate(train, threshold, correction_penalty)
    if cur[:estimated_cost] < best_margin_train[:estimated_cost] ||
       (cur[:estimated_cost] == best_margin_train[:estimated_cost] && cur[:gated_steps] < best_margin_train[:gated_steps])
      best_margin_train = cur
      best_margin_threshold = threshold
    end
  end
  margin_gate = eval_topk_margin_gate(test, best_margin_threshold, correction_penalty)

  {
    samples:                       samples.size,
    train_samples:                 train.size,
    test_samples:                  test.size,
    best_token_scale:              best_token_scale,
    best_rank_scale:               best_rank_scale,
    best_margin_threshold:         best_margin_threshold,
    train_top1_rate:               best_train[:top1_rate],
    train_topk_rate:               best_train[:topk_rate],
    train_avg_branch_tokens:       best_train[:avg_branch_tokens],
    baseline_top1_rate:            baseline[:top1_rate],
    baseline_topk_rate:            baseline[:topk_rate],
    baseline_avg_branch_tokens:    baseline[:avg_branch_tokens],
    calibrated_top1_rate:          calibrated[:top1_rate],
    calibrated_topk_rate:          calibrated[:topk_rate],
    calibrated_avg_branch_tokens:  calibrated[:avg_branch_tokens],
    margin_gate_rate:              margin_gate[:gate_rate],
    margin_gate_topk_rate:         margin_gate[:topk_rate],
    margin_gate_avg_branch_tokens: margin_gate[:avg_branch_tokens],
    margin_gate_misses:            margin_gate[:misses],
    margin_gate_cost:              margin_gate[:estimated_cost],
    baseline_misses:               baseline[:misses],
    calibrated_misses:             calibrated[:misses],
    exact_ids:                     exact_ids,
  }
end

private def print_block_surrogate_topk_oracle(prefix : String,
                                              oracle,
                                              top_k : Int32,
                                              gen_tokens : Int32) : Nil
  delta_branch = oracle[:baseline_avg_branch_tokens] - oracle[:calibrated_avg_branch_tokens]
  puts "#{prefix} top_k=#{top_k} gen_tokens=#{gen_tokens} samples=#{oracle[:samples]} train=#{oracle[:train_samples]} test=#{oracle[:test_samples]} best_token_scale=#{oracle[:best_token_scale]} best_rank_scale=#{oracle[:best_rank_scale]} best_margin_threshold=#{oracle[:best_margin_threshold].round(4)} train_top1=#{oracle[:train_top1_rate].round(2)}% train_topk=#{oracle[:train_topk_rate].round(2)}% train_avg_branch=#{oracle[:train_avg_branch_tokens].round(3)} baseline_top1=#{oracle[:baseline_top1_rate].round(2)}% baseline_topk=#{oracle[:baseline_topk_rate].round(2)}% baseline_avg_branch=#{oracle[:baseline_avg_branch_tokens].round(3)} baseline_misses=#{oracle[:baseline_misses]} calibrated_top1=#{oracle[:calibrated_top1_rate].round(2)}% calibrated_topk=#{oracle[:calibrated_topk_rate].round(2)}% calibrated_avg_branch=#{oracle[:calibrated_avg_branch_tokens].round(3)} calibrated_misses=#{oracle[:calibrated_misses]} delta_avg_branch=#{delta_branch.round(3)} margin_gate_rate=#{oracle[:margin_gate_rate].round(2)}% margin_gate_topk=#{oracle[:margin_gate_topk_rate].round(2)}% margin_gate_avg_branch=#{oracle[:margin_gate_avg_branch_tokens].round(3)} margin_gate_misses=#{oracle[:margin_gate_misses]} margin_gate_cost=#{oracle[:margin_gate_cost].round(3)} exact_ids=#{oracle[:exact_ids].join(',')}"
end

private def append_block_surrogate_suite_rows(rows : Array(BlockSurrogateSuiteRow),
                                             weights,
                                             prompt_name : String,
                                             token_ids : Array(Int32),
                                             block_start : Int32,
                                             block_end : Int32,
                                             adapter,
                                             stats,
                                             mode : String,
                                             rank : Int32,
                                             clusters : Int32,
                                             calib_count : Int32,
                                             gen_tokens : Int32,
                                             gammas : Array(Int32),
                                             state_mode : String,
                                             tree_rows : Array(BlockSurrogateTreeSuiteRow),
                                             tree_top_k : Int32?,
                                             tree_warmup_tokens : Int32,
                                             tree_prefill_seed : Bool,
                                             tree_branch_verify : Bool,
                                             tree_select_advance : Bool,
                                             topk_oracle_k : Int32?,
                                             topk_oracle_train_tokens : Int32?,
                                             min_ideal_speedup : Float64) : Nil
  block = layer_block_label(block_start, block_end)
  gammas.each do |gamma|
    spec = simulate_block_surrogate_self_spec_policy(weights, token_ids, gen_tokens, gamma,
      block_start, block_end, adapter, calib_count, state_mode)
    draft_margin_min = spec[:draft_min_margin_history].empty? ? 0.0 : spec[:draft_min_margin_history].min
    economics = (spec[:parity] && spec[:verifier_parity] && spec[:ideal_overlap_speedup] >= min_ideal_speedup) ? "candidate" : "fail_closed"
    puts "block_surrogate_suite_self_spec prompt=#{prompt_name} block=#{block} mode=#{mode} state_mode=#{state_mode} rank=#{rank} clusters=#{clusters} gamma=#{gamma} gen_tokens=#{gen_tokens} chunks=#{spec[:chunks]} full_accept_chunks=#{spec[:full_accept_chunks]} rejections=#{spec[:rejections]} accepted_draft_tokens=#{spec[:accepted_draft_tokens]} proposed_tokens=#{spec[:proposed_tokens]} accept_rate=#{spec[:accept_rate].round(2)}% avg_accept=#{spec[:avg_accept].round(3)} verifier_tokens=#{spec[:verifier_tokens]} correction_steps=#{spec[:correction_steps]} draft_top2_hit=#{spec[:draft_top2_hit_rate].round(2)}% draft_top5_hit=#{spec[:draft_top5_hit_rate].round(2)}% draft_margin_min=#{draft_margin_min.round(4)} baseline_decode_ms=#{spec[:baseline_decode_ms].round(3)} draft_ms=#{spec[:draft_ms].round(3)} draft_fork_ms=#{spec[:draft_fork_ms].round(3)} verifier_ms=#{spec[:verifier_ms].round(3)} verifier_fork_ms=#{spec[:verifier_fork_ms].round(3)} self_seq_decode_ms=#{spec[:self_seq_decode_ms].round(3)} ideal_overlap_decode_ms=#{spec[:ideal_overlap_decode_ms].round(3)} cpu_seq_speedup=#{spec[:cpu_seq_speedup].round(4)} ideal_overlap_speedup=#{spec[:ideal_overlap_speedup].round(4)} economics=#{economics} min_ideal_speedup=#{min_ideal_speedup.round(4)} parity=#{spec[:parity]} verifier_parity=#{spec[:verifier_parity]} gamma_history=#{spec[:gamma_history].join(',')} accept_history=#{spec[:accept_history].join(',')}"
    rows << {
      prompt:                prompt_name,
      block:                 block,
      mode:                  mode,
      rank:                  rank,
      gamma:                 gamma,
      parity:                spec[:parity],
      verifier_parity:       spec[:verifier_parity],
      accept_rate:           spec[:accept_rate],
      rejections:            spec[:rejections],
      accepted_draft_tokens: spec[:accepted_draft_tokens],
      proposed_tokens:       spec[:proposed_tokens],
      chunks:                spec[:chunks],
      full_accept_chunks:    spec[:full_accept_chunks],
      correction_steps:      spec[:correction_steps],
      draft_top2_hit_rate:   spec[:draft_top2_hit_rate],
      draft_top5_hit_rate:   spec[:draft_top5_hit_rate],
      draft_margin_min:      draft_margin_min,
      baseline_decode_ms:    spec[:baseline_decode_ms],
      draft_ms:              spec[:draft_ms],
      verifier_ms:           spec[:verifier_ms],
      self_seq_decode_ms:    spec[:self_seq_decode_ms],
      ideal_overlap_decode_ms: spec[:ideal_overlap_decode_ms],
      cpu_seq_speedup:       spec[:cpu_seq_speedup],
      ideal_overlap_speedup: spec[:ideal_overlap_speedup],
      hidden_cos_mean:       stats[:mean_cos],
      hidden_cos_min:        stats[:min_cos],
      rel_rmse:              stats[:rel_rmse],
      delta_rel_rmse:        stats[:delta_rel_rmse],
    }
    if top_k = tree_top_k
      tree = simulate_block_surrogate_tree_oracle(weights, token_ids, gen_tokens, top_k, [gamma],
        block_start, block_end, adapter, calib_count, state_mode, tree_warmup_tokens, tree_prefill_seed, tree_branch_verify, tree_select_advance)
      parity = tree[:exact_ids] == tree[:emitted_ids]
      puts "block_surrogate_tree_oracle prompt=#{prompt_name} block=#{block} mode=#{mode} state_mode=#{state_mode} rank=#{rank} clusters=#{clusters} top_k=#{top_k} gamma=#{gamma} gen_tokens=#{gen_tokens} parity=#{parity} #{block_surrogate_tree_metrics_summary(tree)}"
      tree_rows << {
        prompt:                 prompt_name,
        block:                  block,
        mode:                   mode,
        rank:                   rank,
        gamma:                  gamma,
        top_k:                  top_k,
        prefill_seed:           tree[:prefill_seed],
        branch_verify:          tree[:branch_verify],
        select_advance:         tree[:select_advance],
        warmup_tokens:          tree[:warmup_tokens],
        prefill_seed_tokens:    tree[:prefill_seed_tokens],
        tree_tokens:            tree[:tree_tokens],
        parity:                 parity,
        full_rescue_chunks:     tree[:full_rescue_chunks],
        chunks:                 tree[:chunks],
        misses:                 tree[:misses],
        draft_steps:            tree[:draft_steps],
        top1_rate:              tree[:top1_rate],
        topk_rate:              tree[:topk_rate],
        avg_rank_branch_tokens: tree[:avg_rank_branch_tokens],
        avg_full_branch_tokens: tree[:avg_full_branch_tokens],
        avg_rank_branch_tokens_total: tree[:avg_rank_branch_tokens_total],
        avg_full_branch_tokens_total: tree[:avg_full_branch_tokens_total],
        branch_tokens_rank:     tree[:branch_tokens_rank],
        branch_tokens_full:     tree[:branch_tokens_full],
        branch_verify_attempts: tree[:branch_verify_attempts],
        branch_verify_wasted_attempts: tree[:branch_verify_wasted_attempts],
        branch_verify_corrections: tree[:branch_verify_corrections],
        branch_verify_ms:       tree[:branch_verify_ms],
        branch_verify_fork_ms:  tree[:branch_verify_fork_ms],
        branch_verify_forward_ms: tree[:branch_verify_forward_ms],
        correction_steps:       tree[:correction_steps],
        hidden_cos_mean:        stats[:mean_cos],
        rel_rmse:               stats[:rel_rmse],
      }
    end
  end
  if top_k = topk_oracle_k
    oracle = simulate_block_surrogate_topk_oracle_calibration(weights, token_ids, gen_tokens, top_k,
      topk_oracle_train_tokens, block_start, block_end, adapter, calib_count, state_mode)
    print_block_surrogate_topk_oracle("block_surrogate_topk_oracle prompt=#{prompt_name} block=#{block} mode=#{mode} state_mode=#{state_mode} rank=#{rank} clusters=#{clusters}",
      oracle, top_k, gen_tokens)
  end
end

private def block_surrogate_suite_row_score(row : BlockSurrogateSuiteRow) : Float64
  return -1.0e9 unless row[:parity] && row[:verifier_parity]

  row[:accept_rate] - (row[:rejections] * 5.0) + Math.min(row[:draft_margin_min], 10.0) * 0.05
end

private def print_block_surrogate_suite_scoreboard(rows : Array(BlockSurrogateSuiteRow), limit : Int32 = 50) : Nil
  return if rows.empty?

  ranked = rows.sort { |a, b| block_surrogate_suite_row_score(b) <=> block_surrogate_suite_row_score(a) }
  puts "block_surrogate_suite_scoreboard rows=#{rows.size} limit=#{limit}"
  puts "rank prompt block mode gamma parity verifier_parity accept% rejections accepted proposed chunks full_accept margin_min top2% top5% base_ms draft_ms verify_ms seq_ms ideal_ms cpu_x ideal_x hidden_cos rel_rmse delta_rel_rmse score"
  ranked.first(limit).each_with_index do |row, i|
    puts "#{i + 1} #{row[:prompt]} #{row[:block]} #{row[:mode]} #{row[:gamma]} #{row[:parity]} #{row[:verifier_parity]} #{row[:accept_rate].round(2)} #{row[:rejections]} #{row[:accepted_draft_tokens]} #{row[:proposed_tokens]} #{row[:chunks]} #{row[:full_accept_chunks]} #{row[:draft_margin_min].round(4)} #{row[:draft_top2_hit_rate].round(2)} #{row[:draft_top5_hit_rate].round(2)} #{row[:baseline_decode_ms].round(3)} #{row[:draft_ms].round(3)} #{row[:verifier_ms].round(3)} #{row[:self_seq_decode_ms].round(3)} #{row[:ideal_overlap_decode_ms].round(3)} #{row[:cpu_seq_speedup].round(4)} #{row[:ideal_overlap_speedup].round(4)} #{row[:hidden_cos_mean].round(6)} #{row[:rel_rmse].round(6)} #{row[:delta_rel_rmse].round(6)} #{block_surrogate_suite_row_score(row).round(4)}"
  end

  groups = Hash(String, Array(BlockSurrogateSuiteRow)).new { |h, k| h[k] = [] of BlockSurrogateSuiteRow }
  rows.each do |row|
    groups["#{row[:block]}|#{row[:mode]}|#{row[:rank]}|#{row[:gamma]}"] << row
  end

  summaries = [] of NamedTuple(
    key: String,
    prompts: Int32,
    parity_all: Bool,
    verifier_parity_all: Bool,
    accept_mean: Float64,
    accept_min: Float64,
    rejections: Int32,
    accepted: Int32,
    proposed: Int32,
    top2_min: Float64,
    top5_min: Float64,
    margin_min: Float64,
    base_ms_mean: Float64,
    draft_ms_mean: Float64,
    verify_ms_mean: Float64,
    seq_ms_mean: Float64,
    ideal_ms_mean: Float64,
    cpu_seq_speedup_mean: Float64,
    ideal_overlap_speedup_mean: Float64,
    hidden_cos_mean: Float64,
    rel_rmse_mean: Float64,
    delta_rel_rmse_mean: Float64,
    score: Float64)
  groups.each do |key, group|
    prompts = group.size
    parity_all = group.all? { |row| row[:parity] }
    verifier_parity_all = group.all? { |row| row[:verifier_parity] }
    accept_mean = group.sum { |row| row[:accept_rate] } / prompts
    accept_min = group.min_of { |row| row[:accept_rate] }
    rejections = group.sum { |row| row[:rejections] }
    accepted = group.sum { |row| row[:accepted_draft_tokens] }
    proposed = group.sum { |row| row[:proposed_tokens] }
    top2_min = group.min_of { |row| row[:draft_top2_hit_rate] }
    top5_min = group.min_of { |row| row[:draft_top5_hit_rate] }
    margin_min = group.min_of { |row| row[:draft_margin_min] }
    base_ms_mean = group.sum { |row| row[:baseline_decode_ms] } / prompts
    draft_ms_mean = group.sum { |row| row[:draft_ms] } / prompts
    verify_ms_mean = group.sum { |row| row[:verifier_ms] } / prompts
    seq_ms_mean = group.sum { |row| row[:self_seq_decode_ms] } / prompts
    ideal_ms_mean = group.sum { |row| row[:ideal_overlap_decode_ms] } / prompts
    cpu_seq_speedup_mean = group.sum { |row| row[:cpu_seq_speedup] } / prompts
    ideal_overlap_speedup_mean = group.sum { |row| row[:ideal_overlap_speedup] } / prompts
    hidden_cos_mean = group.sum { |row| row[:hidden_cos_mean] } / prompts
    rel_rmse_mean = group.sum { |row| row[:rel_rmse] } / prompts
    delta_rel_rmse_mean = group.sum { |row| row[:delta_rel_rmse] } / prompts
    score = (parity_all && verifier_parity_all) ? (accept_min + accept_mean / 100.0 - rejections * 2.0 + Math.min(margin_min, 10.0) * 0.05) : -1.0e9
    summaries << {
      key:                  key,
      prompts:              prompts,
      parity_all:           parity_all,
      verifier_parity_all:  verifier_parity_all,
      accept_mean:          accept_mean,
      accept_min:           accept_min,
      rejections:           rejections,
      accepted:             accepted,
      proposed:             proposed,
      top2_min:             top2_min,
      top5_min:             top5_min,
      margin_min:           margin_min,
      base_ms_mean:         base_ms_mean,
      draft_ms_mean:        draft_ms_mean,
      verify_ms_mean:       verify_ms_mean,
      seq_ms_mean:          seq_ms_mean,
      ideal_ms_mean:        ideal_ms_mean,
      cpu_seq_speedup_mean: cpu_seq_speedup_mean,
      ideal_overlap_speedup_mean: ideal_overlap_speedup_mean,
      hidden_cos_mean:      hidden_cos_mean,
      rel_rmse_mean:        rel_rmse_mean,
      delta_rel_rmse_mean:  delta_rel_rmse_mean,
      score:                score,
    }
  end

  ranked_summaries = summaries.sort { |a, b| b[:score] <=> a[:score] }
  puts "block_surrogate_suite_aggregate groups=#{summaries.size} limit=#{limit}"
  puts "rank block mode adapter_rank gamma prompts parity_all verifier_parity_all accept_mean accept_min rejections accepted proposed top2_min top5_min margin_min base_ms draft_ms verify_ms seq_ms ideal_ms cpu_x ideal_x hidden_cos_mean rel_rmse_mean delta_rel_rmse_mean score"
  ranked_summaries.first(limit).each_with_index do |row, i|
    block, mode, adapter_rank, gamma = row[:key].split('|')
    puts "#{i + 1} #{block} #{mode} #{adapter_rank} #{gamma} #{row[:prompts]} #{row[:parity_all]} #{row[:verifier_parity_all]} #{row[:accept_mean].round(2)} #{row[:accept_min].round(2)} #{row[:rejections]} #{row[:accepted]} #{row[:proposed]} #{row[:top2_min].round(2)} #{row[:top5_min].round(2)} #{row[:margin_min].round(4)} #{row[:base_ms_mean].round(3)} #{row[:draft_ms_mean].round(3)} #{row[:verify_ms_mean].round(3)} #{row[:seq_ms_mean].round(3)} #{row[:ideal_ms_mean].round(3)} #{row[:cpu_seq_speedup_mean].round(4)} #{row[:ideal_overlap_speedup_mean].round(4)} #{row[:hidden_cos_mean].round(6)} #{row[:rel_rmse_mean].round(6)} #{row[:delta_rel_rmse_mean].round(6)} #{row[:score].round(4)}"
  end
end

private def print_block_surrogate_tree_scoreboard(rows : Array(BlockSurrogateTreeSuiteRow), limit : Int32 = 50) : Nil
  return if rows.empty?

  groups = Hash(String, Array(BlockSurrogateTreeSuiteRow)).new { |h, k| h[k] = [] of BlockSurrogateTreeSuiteRow }
  rows.each do |row|
    groups["#{row[:block]}|#{row[:mode]}|#{row[:rank]}|#{row[:gamma]}|#{row[:top_k]}|#{row[:prefill_seed]}|#{row[:branch_verify]}|#{row[:select_advance]}|#{row[:warmup_tokens]}"] << row
  end

  summaries = [] of NamedTuple(
    key: String,
    prompts: Int32,
    parity_all: Bool,
    prefill_seed_tokens: Int32,
    tree_tokens: Int32,
    top1_mean: Float64,
    top1_min: Float64,
    topk_mean: Float64,
    topk_min: Float64,
    misses: Int32,
    correction_steps: Int32,
    draft_steps: Int32,
    full_rescue_chunks: Int32,
    chunks: Int32,
    avg_rank_branch_tokens: Float64,
    avg_full_branch_tokens: Float64,
    avg_rank_branch_tokens_total: Float64,
    avg_full_branch_tokens_total: Float64,
    branch_verify_attempts: Int32,
    branch_verify_wasted_attempts: Int32,
    branch_verify_corrections: Int32,
    branch_verify_ms: Float64,
    branch_verify_fork_ms: Float64,
    branch_verify_forward_ms: Float64,
    hidden_cos_mean: Float64,
    rel_rmse_mean: Float64,
    score: Float64)
  groups.each do |key, group|
    prompts = group.size
    parity_all = group.all? { |row| row[:parity] }
    top1_mean = group.sum { |row| row[:top1_rate] } / prompts
    top1_min = group.min_of { |row| row[:top1_rate] }
    topk_mean = group.sum { |row| row[:topk_rate] } / prompts
    topk_min = group.min_of { |row| row[:topk_rate] }
    misses = group.sum { |row| row[:misses] }
    correction_steps = group.sum { |row| row[:correction_steps] }
    draft_steps = group.sum { |row| row[:draft_steps] }
    prefill_seed_tokens = group.sum { |row| row[:prefill_seed_tokens] }
    tree_tokens = group.sum { |row| row[:tree_tokens] }
    full_rescue_chunks = group.sum { |row| row[:full_rescue_chunks] }
    chunks = group.sum { |row| row[:chunks] }
    avg_rank_branch_tokens = group.sum { |row| row[:avg_rank_branch_tokens] } / prompts
    avg_full_branch_tokens = group.sum { |row| row[:avg_full_branch_tokens] } / prompts
    avg_rank_branch_tokens_total = group.sum { |row| row[:avg_rank_branch_tokens_total] } / prompts
    avg_full_branch_tokens_total = group.sum { |row| row[:avg_full_branch_tokens_total] } / prompts
    branch_verify_attempts = group.sum { |row| row[:branch_verify_attempts] }
    branch_verify_wasted_attempts = group.sum { |row| row[:branch_verify_wasted_attempts] }
    branch_verify_corrections = group.sum { |row| row[:branch_verify_corrections] }
    branch_verify_ms = group.sum { |row| row[:branch_verify_ms] }
    branch_verify_fork_ms = group.sum { |row| row[:branch_verify_fork_ms] }
    branch_verify_forward_ms = group.sum { |row| row[:branch_verify_forward_ms] }
    hidden_cos_mean = group.sum { |row| row[:hidden_cos_mean] } / prompts
    rel_rmse_mean = group.sum { |row| row[:rel_rmse] } / prompts
    score = parity_all ? (topk_min + topk_mean / 100.0 - misses * 3.0 - avg_rank_branch_tokens * 0.5) : -1.0e9
    summaries << {
      key:                    key,
      prompts:                prompts,
      parity_all:             parity_all,
      prefill_seed_tokens:    prefill_seed_tokens,
      tree_tokens:            tree_tokens,
      top1_mean:              top1_mean,
      top1_min:               top1_min,
      topk_mean:              topk_mean,
      topk_min:               topk_min,
      misses:                 misses,
      correction_steps:       correction_steps,
      draft_steps:            draft_steps,
      full_rescue_chunks:     full_rescue_chunks,
      chunks:                 chunks,
      avg_rank_branch_tokens: avg_rank_branch_tokens,
      avg_full_branch_tokens: avg_full_branch_tokens,
      avg_rank_branch_tokens_total: avg_rank_branch_tokens_total,
      avg_full_branch_tokens_total: avg_full_branch_tokens_total,
      branch_verify_attempts: branch_verify_attempts,
      branch_verify_wasted_attempts: branch_verify_wasted_attempts,
      branch_verify_corrections: branch_verify_corrections,
      branch_verify_ms:       branch_verify_ms,
      branch_verify_fork_ms:  branch_verify_fork_ms,
      branch_verify_forward_ms: branch_verify_forward_ms,
      hidden_cos_mean:        hidden_cos_mean,
      rel_rmse_mean:          rel_rmse_mean,
      score:                  score,
    }
  end

  ranked = summaries.sort { |a, b| b[:score] <=> a[:score] }
  puts "block_surrogate_tree_aggregate groups=#{summaries.size} limit=#{limit}"
  puts "rank block mode adapter_rank gamma top_k prefill_seed branch_verify select_advance warmup prompts parity_all prefill_seed_tokens tree_tokens top1_mean top1_min topk_mean topk_min misses correction_steps draft_steps full_rescue_chunks chunks avg_rank_branch_tokens avg_full_branch_tokens avg_rank_branch_tokens_total avg_full_branch_tokens_total branch_verify_attempts branch_verify_wasted_attempts branch_verify_corrections branch_verify_ms branch_verify_fork_ms branch_verify_forward_ms hidden_cos_mean rel_rmse_mean score"
  ranked.first(limit).each_with_index do |row, i|
    block, mode, adapter_rank, gamma, top_k, prefill_seed, branch_verify, select_advance, warmup = row[:key].split('|')
    puts "#{i + 1} #{block} #{mode} #{adapter_rank} #{gamma} #{top_k} #{prefill_seed} #{branch_verify} #{select_advance} #{warmup} #{row[:prompts]} #{row[:parity_all]} #{row[:prefill_seed_tokens]} #{row[:tree_tokens]} #{row[:top1_mean].round(2)} #{row[:top1_min].round(2)} #{row[:topk_mean].round(2)} #{row[:topk_min].round(2)} #{row[:misses]} #{row[:correction_steps]} #{row[:draft_steps]} #{row[:full_rescue_chunks]} #{row[:chunks]} #{row[:avg_rank_branch_tokens].round(3)} #{row[:avg_full_branch_tokens].round(3)} #{row[:avg_rank_branch_tokens_total].round(3)} #{row[:avg_full_branch_tokens_total].round(3)} #{row[:branch_verify_attempts]} #{row[:branch_verify_wasted_attempts]} #{row[:branch_verify_corrections]} #{row[:branch_verify_ms].round(3)} #{row[:branch_verify_fork_ms].round(3)} #{row[:branch_verify_forward_ms].round(3)} #{row[:hidden_cos_mean].round(6)} #{row[:rel_rmse_mean].round(6)} #{row[:score].round(4)}"
  end
end

private def run_block_surrogate_suite(weights : ML::GGUF::Qwen35Weights,
                                      token_sets : Array(PromptTokenSet),
                                      blocks : Array(LayerBlock),
                                      block_rank : Int32,
                                      pca_iters : Int32,
                                      calib_tokens : Int32,
                                      gen_tokens : Int32,
                                      gammas : Array(Int32),
                                      cluster_count : Int32,
                                      delta_basis_modes : Array(String),
                                      state_mode : String,
                                      oracle_gen_calib : Int32,
                                      tree_top_k : Int32?,
                                      tree_warmup_tokens : Int32,
                                      tree_prefill_seed : Bool,
                                      tree_branch_verify : Bool,
                                      tree_select_advance : Bool,
                                      topk_oracle_k : Int32?,
                                      topk_oracle_train_tokens : Int32?,
                                      min_ideal_speedup : Float64) : Array(BlockSurrogateSuiteRow)
  rows = [] of BlockSurrogateSuiteRow
  tree_rows = [] of BlockSurrogateTreeSuiteRow
  token_sets.each do |prompt_case|
    prompt_name = prompt_case[:name]
    ids = prompt_case[:token_ids]
    prompt_calib_count = Math.min(calib_tokens, ids.size - 1)
    raise "block surrogate suite prompt #{prompt_name.inspect} needs at least one held-out token" unless prompt_calib_count > 0 && prompt_calib_count < ids.size
    oracle_ids = oracle_gen_calib > 0 ? exact_greedy_generated_ids(weights, ids, oracle_gen_calib) : [] of Int32
    sample_ids = ids + oracle_ids
    train_count = Math.min(prompt_calib_count + oracle_ids.size, sample_ids.size - 1)
    puts "block_surrogate_suite_prompt name=#{prompt_name} token_vectors=#{ids.size} calib_tokens=#{prompt_calib_count} heldout_tokens=#{ids.size - prompt_calib_count} oracle_gen_calib=#{oracle_ids.size} train_samples=#{train_count} sample_vectors=#{sample_ids.size} blocks=#{blocks.map { |b| layer_block_label(b[:start], b[:end]) }.join(',')} gammas=#{gammas.join(',')}"

    blocks.each do |block|
      block_start = block[:start]
      block_end = block[:end]
      raise "block surrogate suite end must be within layer count" unless block_end < weights.layers.size
      t0 = Time.instant
      samples = collect_block_residual_samples(weights, sample_ids, block_start, block_end)
      collect_ms = (Time.instant - t0).total_milliseconds
      train_samples = samples[0, train_count]
      impact_basis = if delta_basis_modes.any? { |mode| mode != "pca" }
                       output_margin_impact_vectors(weights, sample_ids[0, train_count])
                     else
                       [] of Array(Float64)
                     end

      block_label = layer_block_label(block_start, block_end)
      delta_basis_modes.each do |delta_basis_mode|
        t_train = Time.instant
        adapter = train_block_residual_surrogate(train_samples, block_start, block_end, block_rank, pca_iters,
          delta_basis_mode: delta_basis_mode, impact_basis_seed: impact_basis)
        train_ms = (Time.instant - t_train).total_milliseconds
        stats = block_residual_surrogate_stats(samples, adapter, train_count)
        mode_label = delta_basis_mode == "pca" ? "global" : "global_#{delta_basis_mode}"
        puts "block_residual_surrogate_suite_static prompt=#{prompt_name} block=#{block_label} mode=#{mode_label} delta_basis=#{delta_basis_mode} impact_vectors=#{impact_basis.size} rank=#{block_rank} effective_input_rank=#{adapter.input_basis.size} effective_delta_rank=#{adapter.delta_basis.size} calib=#{prompt_calib_count} oracle_gen_calib=#{oracle_ids.size} train_samples=#{train_count} heldout=#{stats[:count]} hidden_cos_mean=#{stats[:mean_cos].round(8)} hidden_cos_min=#{stats[:min_cos].round(8)} delta_cos_mean=#{stats[:mean_delta_cos].round(8)} rel_rmse=#{stats[:rel_rmse].round(8)} delta_rel_rmse=#{stats[:delta_rel_rmse].round(8)} adapter_ms=#{stats[:adapter_ms].round(3)} adapter_ms_per_sample=#{stats[:adapter_ms_per_sample].round(6)} collect_ms=#{collect_ms.round(3)} train_ms=#{train_ms.round(3)}"
        append_block_surrogate_suite_rows(rows, weights, prompt_name, ids, block_start, block_end,
          adapter, stats, mode_label, block_rank, 1, prompt_calib_count, gen_tokens, gammas, state_mode,
          tree_rows, tree_top_k, tree_warmup_tokens, tree_prefill_seed, tree_branch_verify, tree_select_advance,
          topk_oracle_k, topk_oracle_train_tokens, min_ideal_speedup)
      end

      next unless cluster_count > 1

      t_mix = Time.instant
      mixture = train_block_residual_mixture(train_samples, block_start, block_end, block_rank, cluster_count, pca_iters)
      mix_train_ms = (Time.instant - t_mix).total_milliseconds
      mix_stats = block_residual_mixture_stats(samples, mixture, train_count)
      puts "block_residual_surrogate_suite_static prompt=#{prompt_name} block=#{block_label} mode=mixture rank=#{block_rank} clusters=#{mixture.centroids.size} cluster_sizes=#{mixture.cluster_sizes.join(',')} calib=#{prompt_calib_count} oracle_gen_calib=#{oracle_ids.size} train_samples=#{train_count} heldout=#{mix_stats[:count]} hidden_cos_mean=#{mix_stats[:mean_cos].round(8)} hidden_cos_min=#{mix_stats[:min_cos].round(8)} delta_cos_mean=#{mix_stats[:mean_delta_cos].round(8)} rel_rmse=#{mix_stats[:rel_rmse].round(8)} delta_rel_rmse=#{mix_stats[:delta_rel_rmse].round(8)} adapter_ms=#{mix_stats[:adapter_ms].round(3)} adapter_ms_per_sample=#{mix_stats[:adapter_ms_per_sample].round(6)} train_ms=#{mix_train_ms.round(3)}"
      append_block_surrogate_suite_rows(rows, weights, prompt_name, ids, block_start, block_end,
        mixture, mix_stats, "mixture", block_rank, mixture.centroids.size, prompt_calib_count, gen_tokens,
        gammas, state_mode, tree_rows, tree_top_k, tree_warmup_tokens, tree_prefill_seed, tree_branch_verify,
        tree_select_advance, topk_oracle_k, topk_oracle_train_tokens, min_ideal_speedup)
    end
  end

  print_block_surrogate_suite_scoreboard(rows)
  print_block_surrogate_tree_scoreboard(tree_rows)
  rows
end

private def simulate_self_spec_policy(weights : ML::GGUF::Qwen35Weights,
                                      prompt_ids : Array(Int32),
                                      gen_tokens : Int32,
                                      gamma : Int32,
                                      layer_bases : LayerBasisMap,
                                      rank : Int32,
                                      calib_count : Int32,
                                      fallback_threshold : Float64?,
                                      refresh_interval : Int32?,
                                      adaptive_min_gamma : Int32? = nil,
                                      adaptive_max_gamma : Int32? = nil,
                                      adaptive_grow_margin_threshold : Float64? = nil,
                                      draft_margin_threshold : Float64? = nil,
                                      draft_stop_margin_threshold : Float64? = nil,
                                      topk_rescue : Int32? = nil,
                                      progressive_schedule : Array(Int32)? = nil) : NamedTuple(chunks: Int32, full_accept_chunks: Int32, rejections: Int32, topk_rescues: Int32, emitted_tokens: Int32, proposed_tokens: Int32, accepted_draft_tokens: Int32, verifier_tokens: Int32, correction_steps: Int32, approx_steps: Int32, fallback_steps: Int32, draft_top2_hits: Int32, draft_top5_hits: Int32, reject_top2_hits: Int32, reject_top5_hits: Int32, accept_rate: Float64, avg_accept: Float64, break_even_draft_verify_per_proposed: Float64, draft_top2_hit_rate: Float64, draft_top5_hit_rate: Float64, gamma_history: Array(Int32), verifier_history: Array(Int32), draft_min_margin_history: Array(Float64), draft_low_margin_history: Array(Int32), exact_ids: Array(Int32), emitted_ids: Array(Int32))
  raise "self-spec gamma must be positive" unless gamma > 0
  adaptive = !adaptive_min_gamma.nil? && !adaptive_max_gamma.nil?
  progressive = progressive_schedule && !progressive_schedule.not_nil!.empty?
  min_gamma = adaptive_min_gamma || gamma
  max_gamma = adaptive_max_gamma || (progressive ? progressive_schedule.not_nil!.max : gamma)
  raise "adaptive min gamma must be positive" if adaptive && min_gamma <= 0
  raise "adaptive max gamma must be >= min gamma" if adaptive && max_gamma < min_gamma
  raise "progressive schedule values must be positive" if progressive && progressive_schedule.not_nil!.any? { |v| v <= 0 }

  hp = weights.hparams
  max_seq = prompt_ids.size + gen_tokens + max_gamma + 4
  exact_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  exact_lr_states = {} of Int32 => LowRankState
  exact_logits = [] of Float32
  state_before_last = exact_state.fork
  last_token = prompt_ids[0]
  pos_last = 0

  prompt_ids.each_with_index do |token_id, pos|
    state_before_last = exact_state.fork
    last_token = token_id
    pos_last = pos
    exact_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
  end

  chunks = 0
  full_accept_chunks = 0
  rejections = 0
  topk_rescues = 0
  emitted_tokens = 0
  proposed_tokens = 0
  accepted_draft_tokens = 0
  verifier_tokens = 0
  correction_steps = 0
  approx_steps = 0
  fallback_steps = 0
  draft_top2_hits = 0
  draft_top5_hits = 0
  reject_top2_hits = 0
  reject_top5_hits = 0
  current_gamma = gamma.clamp(min_gamma, max_gamma)
  progressive_index = 0
  gamma_history = [] of Int32
  verifier_history = [] of Int32
  draft_min_margin_history = [] of Float64
  draft_low_margin_history = [] of Int32
  exact_ids = [] of Int32
  emitted_ids = [] of Int32

  while emitted_tokens < gen_tokens
    chunks += 1
    remaining = gen_tokens - emitted_tokens
    chunk_gamma = if progressive
                    progressive_schedule.not_nil![progressive_index]
                  else
                    current_gamma
                  end
    chunk_gamma = Math.min(chunk_gamma, remaining)
    gamma_history << chunk_gamma

    draft_state = state_before_last.fork
    draft_lr_states = {} of Int32 => LowRankState
    sync_lowrank_shadow!(draft_state, state_before_last, layer_bases, draft_lr_states, rank, hp)
    draft_logits = logits_with_lowrank_policy(weights, last_token, pos_last.to_i32, draft_state,
      layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)

    proposal = [] of Int32
    proposal_top5 = [] of Array(Int32)
    chunk_draft_min_margin = Float64::INFINITY
    chunk_draft_low_margin = 0
    chunk_gamma.times do |j|
      draft_margin = top1_margin(draft_logits)
      chunk_draft_min_margin = draft_margin if draft_margin < chunk_draft_min_margin
      if threshold = draft_margin_threshold
        chunk_draft_low_margin += 1 if draft_margin < threshold
      end
      draft_top5 = top_k_indices(draft_logits, 5)
      proposed = top1(draft_logits)
      proposal << proposed
      proposal_top5 << draft_top5
      if threshold = draft_stop_margin_threshold
        break if proposal.size >= Math.min(min_gamma, chunk_gamma) && draft_margin < threshold
      end
      break if j == chunk_gamma - 1

      draft_logits = logits_with_lowrank_policy(weights, proposed, (pos_last + 1 + j).to_i32, draft_state,
        layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)
    end
    draft_min_margin_history << chunk_draft_min_margin
    draft_low_margin_history << chunk_draft_low_margin
    approx_steps += draft_lr_states.values.sum(&.approx_steps)
    fallback_steps += draft_lr_states.values.sum(&.fallback_steps)
    proposed_tokens += proposal.size
    verifier_tokens += proposal.size
    verifier_history << proposal.size

    accepted_this_chunk = 0
    rejected = false
    chunk_min_margin = Float64::INFINITY
    proposal.each_with_index do |draft_token, j|
      exact_top1 = top1(exact_logits)
      exact_margin = top1_margin(exact_logits)
      chunk_min_margin = exact_margin if exact_margin < chunk_min_margin
      exact_ids << exact_top1
      top5 = proposal_top5[j]
      top2_hit = top5[0, 2].includes?(exact_top1)
      top5_hit = top5.includes?(exact_top1)
      draft_top2_hits += 1 if top2_hit
      draft_top5_hits += 1 if top5_hit
      if draft_token == exact_top1
        accepted_this_chunk += 1
        accepted_draft_tokens += 1
        emitted_tokens += 1
        emitted_ids << draft_token
        state_before_last = exact_state.fork
        last_token = draft_token
        pos_last = prompt_ids.size + emitted_tokens - 1
        exact_logits = logits_with_lowrank_policy(weights, draft_token, pos_last.to_i32, exact_state,
          layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
      else
        rejections += 1
        rescue_hit = false
        if k = topk_rescue
          if k > 1
            rescue_hit = top5[0, Math.min(k, top5.size)].includes?(exact_top1)
          end
        end
        if rescue_hit
          topk_rescues += 1
        else
          correction_steps += 1
        end
        emitted_tokens += 1
        emitted_ids << exact_top1
        state_before_last = exact_state.fork
        last_token = exact_top1
        pos_last = prompt_ids.size + emitted_tokens - 1
        exact_logits = logits_with_lowrank_policy(weights, exact_top1, pos_last.to_i32, exact_state,
          layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
        reject_top2_hits += 1 if top2_hit
        reject_top5_hits += 1 if top5_hit
        rejected = true
        break
      end
      break if emitted_tokens >= gen_tokens
    end
    full_accept_chunks += 1 unless rejected
    # If the final chunk was shorter than gamma and fully accepted, this is still
    # a full accept for the proposed chunk length.
    if adaptive
      current_gamma = if rejected
                        Math.max(min_gamma, current_gamma // 2)
                      elsif threshold = adaptive_grow_margin_threshold
                        chunk_min_margin >= threshold ? Math.min(max_gamma, current_gamma * 2) : current_gamma
                      else
                        Math.min(max_gamma, current_gamma * 2)
                      end
    elsif progressive
      progressive_index = rejected ? 0 : ((progressive_index + 1) % progressive_schedule.not_nil!.size)
    end
  end

  accept_rate = proposed_tokens > 0 ? (100.0 * accepted_draft_tokens / proposed_tokens) : 0.0
  avg_accept = chunks > 0 ? (accepted_draft_tokens.to_f64 / chunks) : 0.0
  draft_top2_hit_rate = proposed_tokens > 0 ? (100.0 * draft_top2_hits / proposed_tokens) : 0.0
  draft_top5_hit_rate = proposed_tokens > 0 ? (100.0 * draft_top5_hits / proposed_tokens) : 0.0
  # Normalized to one exact sequential target decode per emitted token. Correction
  # steps consume exact target work outside the chunk verifier, so the remaining
  # budget is what the low-rank draft plus chunk verifier may spend per proposal.
  break_even = proposed_tokens > 0 ? ((gen_tokens - correction_steps).to_f64 / proposed_tokens) : 0.0

  {
    chunks:                               chunks,
    full_accept_chunks:                   full_accept_chunks,
    rejections:                           rejections,
    topk_rescues:                         topk_rescues,
    emitted_tokens:                       emitted_tokens,
    proposed_tokens:                      proposed_tokens,
    accepted_draft_tokens:                accepted_draft_tokens,
    verifier_tokens:                      verifier_tokens,
    correction_steps:                     correction_steps,
    approx_steps:                         approx_steps,
    fallback_steps:                       fallback_steps,
    draft_top2_hits:                      draft_top2_hits,
    draft_top5_hits:                      draft_top5_hits,
    reject_top2_hits:                     reject_top2_hits,
    reject_top5_hits:                     reject_top5_hits,
    accept_rate:                          accept_rate,
    avg_accept:                           avg_accept,
    break_even_draft_verify_per_proposed: break_even,
    draft_top2_hit_rate:                  draft_top2_hit_rate,
    draft_top5_hit_rate:                  draft_top5_hit_rate,
    gamma_history:                        gamma_history,
    verifier_history:                     verifier_history,
    draft_min_margin_history:             draft_min_margin_history,
    draft_low_margin_history:             draft_low_margin_history,
    exact_ids:                            exact_ids,
    emitted_ids:                          emitted_ids,
  }
end

private def simulate_self_spec_tree_oracle(weights : ML::GGUF::Qwen35Weights,
                                           prompt_ids : Array(Int32),
                                           gen_tokens : Int32,
                                           top_k : Int32,
                                           progressive_schedule : Array(Int32),
                                           layer_bases : LayerBasisMap,
                                           rank : Int32,
                                           calib_count : Int32,
                                           fallback_threshold : Float64?,
                                           refresh_interval : Int32?) : NamedTuple(chunks: Int32, full_rescue_chunks: Int32, misses: Int32, emitted_tokens: Int32, draft_steps: Int32, top1_hits: Int32, topk_hits: Int32, branch_tokens_rank: Int32, branch_tokens_full: Int32, correction_steps: Int32, approx_steps: Int32, fallback_steps: Int32, top1_rate: Float64, topk_rate: Float64, avg_rank_branch_tokens: Float64, avg_full_branch_tokens: Float64, schedule_history: Array(Int32), exact_ids: Array(Int32), emitted_ids: Array(Int32))
  raise "tree top_k must be >= 2" unless top_k >= 2
  raise "tree top_k must be <= 16" unless top_k <= 16
  raise "tree progressive schedule must not be empty" if progressive_schedule.empty?
  raise "tree schedule values must be positive" if progressive_schedule.any? { |v| v <= 0 }

  hp = weights.hparams
  max_gamma = progressive_schedule.max
  max_seq = prompt_ids.size + gen_tokens + max_gamma + 4
  exact_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  exact_lr_states = {} of Int32 => LowRankState
  exact_logits = [] of Float32
  state_before_last = exact_state.fork
  last_token = prompt_ids[0]
  pos_last = 0

  prompt_ids.each_with_index do |token_id, pos|
    state_before_last = exact_state.fork
    last_token = token_id
    pos_last = pos
    exact_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
  end

  chunks = 0
  full_rescue_chunks = 0
  misses = 0
  emitted_tokens = 0
  draft_steps = 0
  top1_hits = 0
  topk_hits = 0
  branch_tokens_rank = 0
  branch_tokens_full = 0
  correction_steps = 0
  approx_steps = 0
  fallback_steps = 0
  progressive_index = 0
  schedule_history = [] of Int32
  exact_ids = [] of Int32
  emitted_ids = [] of Int32

  while emitted_tokens < gen_tokens
    chunks += 1
    chunk_gamma = Math.min(progressive_schedule[progressive_index], gen_tokens - emitted_tokens)
    schedule_history << chunk_gamma
    draft_state = state_before_last.fork
    draft_lr_states = {} of Int32 => LowRankState
    sync_lowrank_shadow!(draft_state, state_before_last, layer_bases, draft_lr_states, rank, hp)
    draft_logits = logits_with_lowrank_policy(weights, last_token, pos_last.to_i32, draft_state,
      layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)

    rescued_chunk = true
    chunk_gamma.times do |j|
      exact_top1 = top1(exact_logits)
      exact_ids << exact_top1
      draft_topk = top_k_indices(draft_logits, top_k)
      draft_steps += 1
      if draft_topk[0] == exact_top1
        top1_hits += 1
        topk_hits += 1
        branch_tokens_rank += 1
        branch_tokens_full += 1
      elsif idx = draft_topk.index(exact_top1)
        topk_hits += 1
        branch_tokens_rank += idx + 1
        branch_tokens_full += top_k
      else
        misses += 1
        correction_steps += 1
        branch_tokens_rank += top_k
        branch_tokens_full += top_k
        rescued_chunk = false
      end

      emitted_tokens += 1
      emitted_ids << exact_top1
      state_before_last = exact_state.fork
      last_token = exact_top1
      pos_last = prompt_ids.size + emitted_tokens - 1
      exact_logits = logits_with_lowrank_policy(weights, exact_top1, pos_last.to_i32, exact_state,
        layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
      break if !rescued_chunk || emitted_tokens >= gen_tokens || j == chunk_gamma - 1

      draft_logits = logits_with_lowrank_policy(weights, exact_top1, pos_last.to_i32, draft_state,
        layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)
    end

    approx_steps += draft_lr_states.values.sum(&.approx_steps)
    fallback_steps += draft_lr_states.values.sum(&.fallback_steps)
    full_rescue_chunks += 1 if rescued_chunk
    progressive_index = rescued_chunk ? ((progressive_index + 1) % progressive_schedule.size) : 0
  end

  {
    chunks:                 chunks,
    full_rescue_chunks:     full_rescue_chunks,
    misses:                 misses,
    emitted_tokens:         emitted_tokens,
    draft_steps:            draft_steps,
    top1_hits:              top1_hits,
    topk_hits:              topk_hits,
    branch_tokens_rank:     branch_tokens_rank,
    branch_tokens_full:     branch_tokens_full,
    correction_steps:       correction_steps,
    approx_steps:           approx_steps,
    fallback_steps:         fallback_steps,
    top1_rate:              emitted_tokens > 0 ? 100.0 * top1_hits / emitted_tokens : 0.0,
    topk_rate:              emitted_tokens > 0 ? 100.0 * topk_hits / emitted_tokens : 0.0,
    avg_rank_branch_tokens: emitted_tokens > 0 ? branch_tokens_rank.to_f64 / emitted_tokens : 0.0,
    avg_full_branch_tokens: emitted_tokens > 0 ? branch_tokens_full.to_f64 / emitted_tokens : 0.0,
    schedule_history:       schedule_history,
    exact_ids:              exact_ids,
    emitted_ids:            emitted_ids,
  }
end

private def train_topk_oracle_biases(samples : Array(TopKOracleSample),
                                     top_k : Int32) : NamedTuple(token_bias: Hash(Int32, Float64), rank_bias: Array(Float64))
  token_seen = Hash(Int32, Int32).new(0)
  token_hit = Hash(Int32, Int32).new(0)
  rank_seen = Array(Int32).new(top_k, 0)
  rank_hit = Array(Int32).new(top_k, 0)

  samples.each do |sample|
    sample.ids.each_with_index do |id, rank|
      token_seen[id] += 1
      token_hit[id] += 1 if id == sample.exact_id
      rank_seen[rank] += 1
      rank_hit[rank] += 1 if id == sample.exact_id
    end
  end

  alpha = 0.5
  total_seen = Math.max(1, samples.size * top_k)
  total_hit = samples.count { |s| s.ids.includes?(s.exact_id) }
  global_logit = Math.log((total_hit + alpha) / (total_seen - total_hit + alpha))

  token_bias = {} of Int32 => Float64
  token_seen.each do |id, seen|
    hit = token_hit[id]
    token_bias[id] = Math.log((hit + alpha) / (seen - hit + alpha)) - global_logit
  end

  rank_bias = Array(Float64).new(top_k, 0.0)
  top_k.times do |rank|
    seen = rank_seen[rank]
    hit = rank_hit[rank]
    rank_bias[rank] = seen > 0 ? Math.log((hit + alpha) / (seen - hit + alpha)) - global_logit : 0.0
  end

  {token_bias: token_bias, rank_bias: rank_bias}
end

private def eval_topk_oracle_samples(samples : Array(TopKOracleSample),
                                     token_bias : Hash(Int32, Float64),
                                     rank_bias : Array(Float64),
                                     token_scale : Float64,
                                     rank_scale : Float64) : NamedTuple(samples: Int32, top1_hits: Int32, topk_hits: Int32, misses: Int32, branch_tokens: Int32, top1_rate: Float64, topk_rate: Float64, avg_branch_tokens: Float64)
  top1_hits = 0
  topk_hits = 0
  misses = 0
  branch_tokens = 0

  samples.each do |sample|
    order = (0...sample.ids.size).to_a
    order.sort_by! do |rank|
      id = sample.ids[rank]
      -(sample.logits[rank].to_f64 + token_scale * (token_bias[id]? || 0.0) + rank_scale * rank_bias[rank])
    end

    reranked_ids = order.map { |rank| sample.ids[rank] }
    if reranked_ids[0]? == sample.exact_id
      top1_hits += 1
      topk_hits += 1
      branch_tokens += 1
    elsif idx = reranked_ids.index(sample.exact_id)
      topk_hits += 1
      branch_tokens += idx + 1
    else
      misses += 1
      branch_tokens += sample.ids.size
    end
  end

  n = samples.size
  {
    samples:           n,
    top1_hits:         top1_hits,
    topk_hits:         topk_hits,
    misses:            misses,
    branch_tokens:     branch_tokens,
    top1_rate:         n > 0 ? 100.0 * top1_hits / n : 0.0,
    topk_rate:         n > 0 ? 100.0 * topk_hits / n : 0.0,
    avg_branch_tokens: n > 0 ? branch_tokens.to_f64 / n : 0.0,
  }
end

private def eval_topk_margin_gate(samples : Array(TopKOracleSample),
                                  margin_threshold : Float64,
                                  correction_penalty : Float64) : NamedTuple(samples: Int32, gated_steps: Int32, top1_hits: Int32, topk_hits: Int32, misses: Int32, branch_tokens: Int32, estimated_cost: Float64, gate_rate: Float64, top1_rate: Float64, topk_rate: Float64, avg_branch_tokens: Float64)
  gated_steps = 0
  top1_hits = 0
  topk_hits = 0
  misses = 0
  branch_tokens = 0

  samples.each do |sample|
    exact_rank = sample.ids.index(sample.exact_id)
    if sample.margin < margin_threshold
      gated_steps += 1
      if exact_rank
        topk_hits += 1
        top1_hits += 1 if exact_rank == 0
        branch_tokens += exact_rank + 1
      else
        misses += 1
        branch_tokens += sample.ids.size
      end
    else
      branch_tokens += 1
      if sample.ids[0]? == sample.exact_id
        top1_hits += 1
        topk_hits += 1
      elsif exact_rank
        # The token was available in topK, but the margin gate chose the cheap
        # top1 path, so the verifier would need a correction/resync.
        misses += 1
      else
        misses += 1
      end
    end
  end

  n = samples.size
  {
    samples:           n,
    gated_steps:       gated_steps,
    top1_hits:         top1_hits,
    topk_hits:         topk_hits,
    misses:            misses,
    branch_tokens:     branch_tokens,
    estimated_cost:    branch_tokens.to_f64 + correction_penalty * misses,
    gate_rate:         n > 0 ? 100.0 * gated_steps / n : 0.0,
    top1_rate:         n > 0 ? 100.0 * top1_hits / n : 0.0,
    topk_rate:         n > 0 ? 100.0 * topk_hits / n : 0.0,
    avg_branch_tokens: n > 0 ? branch_tokens.to_f64 / n : 0.0,
  }
end

private def simulate_topk_oracle_calibration(weights : ML::GGUF::Qwen35Weights,
                                             prompt_ids : Array(Int32),
                                             gen_tokens : Int32,
                                             top_k : Int32,
                                             train_tokens : Int32?,
                                             layer_bases : LayerBasisMap,
                                             rank : Int32,
                                             calib_count : Int32,
                                             fallback_threshold : Float64?,
                                             refresh_interval : Int32?) : NamedTuple(samples: Int32, train_samples: Int32, test_samples: Int32, best_token_scale: Float64, best_rank_scale: Float64, best_margin_threshold: Float64, train_top1_rate: Float64, train_topk_rate: Float64, train_avg_branch_tokens: Float64, baseline_top1_rate: Float64, baseline_topk_rate: Float64, baseline_avg_branch_tokens: Float64, calibrated_top1_rate: Float64, calibrated_topk_rate: Float64, calibrated_avg_branch_tokens: Float64, margin_gate_rate: Float64, margin_gate_topk_rate: Float64, margin_gate_avg_branch_tokens: Float64, margin_gate_misses: Int32, margin_gate_cost: Float64, baseline_misses: Int32, calibrated_misses: Int32, exact_ids: Array(Int32))
  raise "topK oracle top_k must be >= 2" unless top_k >= 2
  raise "topK oracle top_k must be <= 16" unless top_k <= 16
  raise "topK oracle gen_tokens must be >= 4" unless gen_tokens >= 4

  hp = weights.hparams
  max_seq = prompt_ids.size + gen_tokens + 4
  exact_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  exact_lr_states = {} of Int32 => LowRankState
  exact_logits = [] of Float32
  state_before_last = exact_state.fork
  last_token = prompt_ids[0]
  pos_last = 0

  prompt_ids.each_with_index do |token_id, pos|
    state_before_last = exact_state.fork
    last_token = token_id
    pos_last = pos
    exact_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, exact_state,
      layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
  end

  draft_state = state_before_last.fork
  draft_lr_states = {} of Int32 => LowRankState
  sync_lowrank_shadow!(draft_state, state_before_last, layer_bases, draft_lr_states, rank, hp)
  draft_logits = logits_with_lowrank_policy(weights, last_token, pos_last.to_i32, draft_state,
    layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)

  samples = [] of TopKOracleSample
  exact_ids = [] of Int32
  gen_tokens.times do |step|
    exact_top1 = top1(exact_logits)
    samples << topk_oracle_sample(draft_logits, exact_top1, top_k)
    exact_ids << exact_top1

    pos_last = prompt_ids.size + step
    exact_logits = logits_with_lowrank_policy(weights, exact_top1, pos_last.to_i32, exact_state,
      layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
    break if step == gen_tokens - 1
    draft_logits = logits_with_lowrank_policy(weights, exact_top1, pos_last.to_i32, draft_state,
      layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true)
  end

  requested_train = train_tokens || (samples.size // 2)
  train_count = requested_train.clamp(1, samples.size - 1)
  train = samples[0, train_count]
  test = samples[train_count, samples.size - train_count]
  biases = train_topk_oracle_biases(train, top_k)
  zero_token_bias = {} of Int32 => Float64
  zero_rank_bias = Array(Float64).new(top_k, 0.0)
  baseline = eval_topk_oracle_samples(test, zero_token_bias, zero_rank_bias, 0.0, 0.0)

  token_scales = [0.0, 0.25, 0.5, 1.0, 2.0]
  rank_scales = [0.0, 0.25, 0.5, 1.0]
  best_token_scale = 0.0
  best_rank_scale = 0.0
  best_train = eval_topk_oracle_samples(train, biases[:token_bias], biases[:rank_bias], 0.0, 0.0)
  token_scales.each do |ts|
    rank_scales.each do |rs|
      cur = eval_topk_oracle_samples(train, biases[:token_bias], biases[:rank_bias], ts, rs)
      if cur[:avg_branch_tokens] < best_train[:avg_branch_tokens] ||
         (cur[:avg_branch_tokens] == best_train[:avg_branch_tokens] && cur[:top1_rate] > best_train[:top1_rate])
        best_train = cur
        best_token_scale = ts
        best_rank_scale = rs
      end
    end
  end

  calibrated = eval_topk_oracle_samples(test, biases[:token_bias], biases[:rank_bias], best_token_scale, best_rank_scale)
  correction_penalty = top_k.to_f64
  thresholds = [-1.0] + train.map(&.margin).uniq.sort + [Float64::INFINITY]
  best_margin_threshold = thresholds[0]
  best_margin_train = eval_topk_margin_gate(train, best_margin_threshold, correction_penalty)
  thresholds.each do |threshold|
    cur = eval_topk_margin_gate(train, threshold, correction_penalty)
    if cur[:estimated_cost] < best_margin_train[:estimated_cost] ||
       (cur[:estimated_cost] == best_margin_train[:estimated_cost] && cur[:gated_steps] < best_margin_train[:gated_steps])
      best_margin_train = cur
      best_margin_threshold = threshold
    end
  end
  margin_gate = eval_topk_margin_gate(test, best_margin_threshold, correction_penalty)
  {
    samples:                       samples.size,
    train_samples:                 train.size,
    test_samples:                  test.size,
    best_token_scale:              best_token_scale,
    best_rank_scale:               best_rank_scale,
    best_margin_threshold:         best_margin_threshold,
    train_top1_rate:               best_train[:top1_rate],
    train_topk_rate:               best_train[:topk_rate],
    train_avg_branch_tokens:       best_train[:avg_branch_tokens],
    baseline_top1_rate:            baseline[:top1_rate],
    baseline_topk_rate:            baseline[:topk_rate],
    baseline_avg_branch_tokens:    baseline[:avg_branch_tokens],
    calibrated_top1_rate:          calibrated[:top1_rate],
    calibrated_topk_rate:          calibrated[:topk_rate],
    calibrated_avg_branch_tokens:  calibrated[:avg_branch_tokens],
    margin_gate_rate:              margin_gate[:gate_rate],
    margin_gate_topk_rate:         margin_gate[:topk_rate],
    margin_gate_avg_branch_tokens: margin_gate[:avg_branch_tokens],
    margin_gate_misses:            margin_gate[:misses],
    margin_gate_cost:              margin_gate[:estimated_cost],
    baseline_misses:               baseline[:misses],
    calibrated_misses:             calibrated[:misses],
    exact_ids:                     exact_ids,
  }
end

private def simulate_self_spec_wall_policy(weights : ML::GGUF::Qwen35Weights,
                                           prompt_ids : Array(Int32),
                                           gen_tokens : Int32,
                                           progressive_schedule : Array(Int32),
                                           layer_bases : LayerBasisMap,
                                           rank : Int32,
                                           calib_count : Int32,
                                           fallback_threshold : Float64?,
                                           refresh_interval : Int32?,
                                           use_metal_lowrank : Bool,
                                           project_coeffs_on_gpu : Bool,
                                           use_metal_layer_updown : Bool = false,
                                           draft_variant : String = "lowrank",
                                           ffn_bases : FFNBasisMap? = nil,
                                           ffn_adapters : FFNAdapterMap? = nil,
                                           ffn_updown_adapters : FFNUpDownAdapterMap? = nil,
                                           ffn_block_selectors : FFNBlockSelectorMap? = nil) : NamedTuple(chunks: Int32, rejections: Int32, accepted_draft_tokens: Int32, proposed_tokens: Int32, verifier_tokens: Int32, correction_steps: Int32, draft_ms: Float64, verifier_ms: Float64, replay_ms: Float64, serial_ms: Float64, overlap_est_ms: Float64, speedup_est: Float64, accept_rate: Float64, exact_ids: Array(Int32), emitted_ids: Array(Int32))
  raise "wall self-spec requires a non-empty progressive schedule" if progressive_schedule.empty?
  raise "wall self-spec schedule values must be positive" if progressive_schedule.any? { |v| v <= 0 }

  hp = weights.hparams
  max_gamma = progressive_schedule.max
  max_seq = prompt_ids.size + gen_tokens + max_gamma + 4

  # CPU shadow state feeds the projected-K draft branch. The verifier state uses
  # the production chunk verifier path, which can route its exact work to Metal.
  shadow_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  verifier_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(verifier_state, hp)

  exact_lr_states = {} of Int32 => LowRankState
  exact_logits = [] of Float32
  state_before_last = shadow_state.fork
  last_token = prompt_ids[0]
  pos_last = 0

  prompt_ids.each_with_index do |token_id, pos|
    state_before_last = shadow_state.fork
    last_token = token_id
    pos_last = pos
    exact_logits = logits_with_lowrank_policy(weights, token_id, pos.to_i32, shadow_state,
      layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
  end
  target_next, _ = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, prompt_ids, 0, verifier_state)
  raise "shadow/verifier prompt top1 mismatch: #{top1(exact_logits)} != #{target_next}" unless top1(exact_logits) == target_next

  emitted_tokens = 0
  chunks = 0
  rejections = 0
  proposed_tokens = 0
  accepted_draft_tokens = 0
  verifier_tokens = 0
  correction_steps = 0
  progressive_index = 0
  target_next_id = target_next.to_i32
  draft_ms = 0.0
  verifier_ms = 0.0
  replay_ms = 0.0
  chunk_draft_ms = [] of Float64
  chunk_verifier_ms = [] of Float64
  exact_ids = [] of Int32
  emitted_ids = [] of Int32

  while emitted_tokens < gen_tokens
    chunks += 1
    chunk_gamma = Math.min(progressive_schedule[progressive_index], gen_tokens - emitted_tokens)

    t_draft = Time.instant
    draft_state = state_before_last.fork
    draft_lr_states = {} of Int32 => LowRankState
    sync_lowrank_shadow!(draft_state, state_before_last, layer_bases, draft_lr_states, rank, hp)
    draft_logits = logits_with_lowrank_policy(weights, last_token, pos_last.to_i32, draft_state,
      layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true, use_metal_lowrank, project_coeffs_on_gpu, use_metal_layer_updown, draft_variant, ffn_bases, ffn_adapters, ffn_updown_adapters, ffn_block_selectors)
    proposal = [] of Int32
    chunk_gamma.times do |j|
      proposed = top1(draft_logits)
      proposal << proposed
      break if j == chunk_gamma - 1
      draft_logits = logits_with_lowrank_policy(weights, proposed, (pos_last + 1 + j).to_i32, draft_state,
        layer_bases, rank, calib_count, draft_lr_states, fallback_threshold, refresh_interval, true, use_metal_lowrank, project_coeffs_on_gpu, use_metal_layer_updown, draft_variant, ffn_bases, ffn_adapters, ffn_updown_adapters, ffn_block_selectors)
    end
    dt_draft = (Time.instant - t_draft).total_milliseconds
    draft_ms += dt_draft
    chunk_draft_ms << dt_draft
    proposed_tokens += proposal.size
    verifier_tokens += proposal.size

    cycle_start_pos = prompt_ids.size + emitted_tokens
    verifier_backup = verifier_state.fork
    t_verify = Time.instant
    target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, proposal, cycle_start_pos, verifier_state)
    dt_verify = (Time.instant - t_verify).total_milliseconds
    verifier_ms += dt_verify
    chunk_verifier_ms << dt_verify

    correction_or_accepted = [] of Int32
    expected = target_next_id
    rejected = false
    proposal.each_with_index do |cand, i|
      exact_ids << expected
      emitted = if cand == expected
                  accepted_draft_tokens += 1
                  cand
                else
                  rejections += 1
                  correction_steps += 1
                  rejected = true
                  expected
                end
      correction_or_accepted << emitted
      emitted_ids << emitted
      emitted_tokens += 1

      pos = cycle_start_pos + i
      state_before_last = shadow_state.fork
      last_token = emitted
      pos_last = pos
      exact_logits = logits_with_lowrank_policy(weights, emitted, pos.to_i32, shadow_state,
        layer_bases, rank, calib_count, exact_lr_states, fallback_threshold, nil, false)
      expected = target_nexts[i][0] if cand == expected
      break if rejected || emitted_tokens >= gen_tokens
    end

    if rejected
      verifier_state.copy_from!(verifier_backup)
      t_replay = Time.instant
      corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, correction_or_accepted, cycle_start_pos, verifier_state)
      replay_ms += (Time.instant - t_replay).total_milliseconds
      target_next_id = corrected[-1][0]
      progressive_index = 0
    else
      target_next_id = target_nexts[correction_or_accepted.size - 1][0]
      progressive_index = (progressive_index + 1) % progressive_schedule.size
    end
  end

  serial_ms = draft_ms + verifier_ms + replay_ms
  overlap_est_ms = 0.0
  chunk_draft_ms.each_with_index do |d_ms, i|
    v_ms = chunk_verifier_ms[i]? || 0.0
    overlap_est_ms += Math.max(d_ms, v_ms)
  end
  overlap_est_ms += replay_ms
  speedup_est = overlap_est_ms > 0.0 ? serial_ms / overlap_est_ms : 0.0
  accept_rate = proposed_tokens > 0 ? (100.0 * accepted_draft_tokens / proposed_tokens) : 0.0

  {
    chunks:                chunks,
    rejections:            rejections,
    accepted_draft_tokens: accepted_draft_tokens,
    proposed_tokens:       proposed_tokens,
    verifier_tokens:       verifier_tokens,
    correction_steps:      correction_steps,
    draft_ms:              draft_ms,
    verifier_ms:           verifier_ms,
    replay_ms:             replay_ms,
    serial_ms:             serial_ms,
    overlap_est_ms:        overlap_est_ms,
    speedup_est:           speedup_est,
    accept_rate:           accept_rate,
    exact_ids:             exact_ids,
    emitted_ids:           emitted_ids,
  }
end

private def parse_int_list(value : String) : Array(Int32)
  value.split(',').map(&.strip).reject(&.empty?).map(&.to_i)
end

private def parse_float_list(value : String) : Array(Float64)
  value.split(',').map(&.strip).reject(&.empty?).map(&.to_f64)
end

private def parse_prefix_suffix_thresholds(value : String) : Array(Tuple(Int32, Float64))
  value.split(',').map(&.strip).reject(&.empty?).map do |raw|
    parts = raw.split(':').map(&.strip)
    raise "prefix/suffix threshold entry expects MIN_PREFIX:THRESHOLD" unless parts.size == 2
    min_prefix = parts[0].to_i
    threshold = parts[1].to_f64
    raise "prefix/suffix threshold min-prefix must be positive" if min_prefix <= 0
    raise "prefix/suffix threshold must be non-negative" if threshold < 0.0
    {min_prefix, threshold}
  end
end

private def apply_branch_snapshot_policy(value : String) : String
  mode = value.strip.downcase
  ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold = nil
  ProbeRuntime.self_spec_branch_guard_snapshot_suffix_min_threshold = nil
  case mode
  when "off", "none", "nosnap"
    ProbeRuntime.self_spec_branch_guard_snapshot = false
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = false
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = false
    "nosnap"
  when "split"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = false
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = false
    "split"
  when "onepass"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = true
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = false
    "onepass"
  when "split_suffix2", "split_min3_suffix2"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = false
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = true
    ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = 3
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = 2.0
    "split_min3_suffix2"
  when "split_min3_suffix2_keepguard"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = false
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = false
    ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = 3
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = 2.0
    "split_min3_suffix2_keepguard"
  when "split_min3_suffix2_guard02"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = false
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = true
    ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = 3
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = 2.0
    ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold = 0.2
    "split_min3_suffix2_guard02"
  when "onepass_suffix2", "onepass_min3_suffix2"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = true
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = true
    ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = 3
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = 2.0
    "onepass_min3_suffix2"
  when "onepass_min3_suffix2_keepguard"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = true
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = false
    ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = 3
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = 2.0
    "onepass_min3_suffix2_keepguard"
  when "onepass_min3_suffix2_guard02"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = true
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = true
    ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = 3
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = 2.0
    ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold = 0.2
    "onepass_min3_suffix2_guard02"
  when "onepass_min3_suffix2_guard005"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = true
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = true
    ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = 3
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = 2.0
    ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold = 0.05
    "onepass_min3_suffix2_guard005"
  when "onepass_min3_suffix2_guard01"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = true
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = true
    ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = 3
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = 2.0
    ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold = 0.1
    "onepass_min3_suffix2_guard01"
  when "onepass_min3_suffix1to2_guard01"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = true
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = true
    ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = 3
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_min_threshold = 1.0
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = 2.0
    ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold = 0.1
    "onepass_min3_suffix1to2_guard01"
  when "onepass_min3_suffix2_guard05"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = true
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = true
    ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = 3
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = 2.0
    ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold = 0.5
    "onepass_min3_suffix2_guard05"
  when "onepass_min3_suffix2_guardinf"
    ProbeRuntime.self_spec_branch_guard_snapshot = true
    ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = true
    ProbeRuntime.self_spec_branch_guard_snapshot_only_split = true
    ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = 3
    ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = 2.0
    ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold = Float64::INFINITY
    "onepass_min3_suffix2_guardinf"
  else
    raise "unknown branch snapshot policy #{value.inspect}; expected off, split, onepass, split_min3_suffix2, onepass_min3_suffix2, *_keepguard, onepass_min3_suffix1to2_guard01, or onepass *_guard005/_guard01/_guard02/_guard05/_guardinf variants"
  end
end

private def parse_layer_block(value : String)
  raw = value.strip
  if raw.includes?(":")
    parts = raw.split(':').map(&.strip)
    raise "layer block expects START:END" unless parts.size == 2
    start_layer = parts[0].to_i
    end_layer = parts[1].to_i
  elsif raw.includes?("..")
    parts = raw.split("..").map(&.strip)
    raise "layer block expects START..END" unless parts.size == 2
    start_layer = parts[0].to_i
    end_layer = parts[1].to_i
  else
    layers = parse_int_list(raw)
    raise "layer block list must not be empty" if layers.empty?
    start_layer = layers.min
    end_layer = layers.max
  end
  raise "layer block start must be <= end" unless start_layer <= end_layer
  {start: start_layer.to_i32, end: end_layer.to_i32}
end

private def parse_layer_block_list(value : String) : Array(LayerBlock)
  blocks = [] of LayerBlock
  value.split(/[,;]/).map(&.strip).reject(&.empty?).each do |raw|
    if raw.includes?("-") && !raw.includes?(":") && !raw.includes?("..")
      parts = raw.split('-').map(&.strip)
      raise "block suite range expects START-END" unless parts.size == 2
      first = parts[0].to_i
      last = parts[1].to_i
      raise "block suite range start must be <= end" unless first <= last
      first.upto(last) { |il| blocks << {start: il.to_i32, end: il.to_i32} }
    else
      block = parse_layer_block(raw)
      blocks << {start: block[:start], end: block[:end]}
    end
  end
  raise "block suite block list must not be empty" if blocks.empty?
  blocks.uniq
end

private def layer_block_label(block_start : Int32, block_end : Int32) : String
  "#{block_start}:#{block_end}"
end

private def cheap_draft_variant_valid?(variant : String) : Bool
  return true if {"lowrank", "lowrank-no-ffn", "skip-layer"}.includes?(variant)
  return true if draft_variant_ffn_top_percent(variant)
  return true if draft_variant_ffn_block_top_percent(variant)
  return true if draft_variant_ffn_block_pred_percent(variant)
  return true if draft_variant_ffn_pca_rank(variant)
  return true if draft_variant_ffn_pca_down_rank(variant)
  return true if draft_variant_ffn_pca_updown_rank(variant)
  return false unless variant.starts_with?("early-exit-")

  variant["early-exit-".size..].to_i? ? true : false
end

private def cheap_draft_early_exit_layers(variant : String) : Int32?
  return nil unless variant.starts_with?("early-exit-")

  n = variant["early-exit-".size..].to_i? || raise "invalid early-exit variant #{variant.inspect}"
  raise "early-exit layer count must be positive" unless n > 0
  n
end

private def self_spec_estimated_cost(spec,
                                     draft_cost : Float64,
                                     verifier_cost : Float64,
                                     chunk_overhead : Float64,
                                     correction_cost : Float64,
                                     overlap : Bool,
                                     overlap_efficiency : Float64) : Float64
  if overlap
    efficiency = overlap_efficiency.clamp(0.0, 1.0)
    cost = 0.0
    spec[:gamma_history].each_with_index do |draft_tokens, i|
      verifier_tokens = spec[:verifier_history][i]
      draft_segment = draft_cost * draft_tokens
      verifier_segment = verifier_cost * verifier_tokens
      hidden = Math.min(draft_segment, verifier_segment) * efficiency
      cost += draft_segment + verifier_segment - hidden + chunk_overhead
    end
    cost + correction_cost * spec[:correction_steps]
  else
    draft_cost * spec[:proposed_tokens] +
      verifier_cost * spec[:verifier_tokens] +
      chunk_overhead * spec[:chunks] +
      correction_cost * spec[:correction_steps]
  end
end

private def self_spec_tree_estimated_cost(tree,
                                          draft_cost : Float64,
                                          verifier_cost : Float64,
                                          chunk_overhead : Float64,
                                          correction_cost : Float64,
                                          branch_tokens : Int32) : Float64
  draft_cost * tree[:draft_steps] +
    verifier_cost * branch_tokens +
    chunk_overhead * tree[:chunks] +
    correction_cost * tree[:correction_steps]
end

private def simulate_self_draft_metal_baseline_run(weights : ML::GGUF::Qwen35Weights,
                                                   token_ids : Array(Int32),
                                                   calib_count : Int32,
                                                   n_draft : Int32,
                                                   layer_bases : Hash(Int32, BasisSet),
                                                   rank : Int32) : NamedTuple(steps: Int32, self_draft_ms: Float64, exact_ms: Float64, verifier_ms: Float64, self_draft_per_token_ms: Float64, exact_per_token_ms: Float64, verifier_per_token_ms: Float64, self_spec_wall_ratio: Float64, agreement: Int32, self_draft_ids: Array(Int32), exact_ids: Array(Int32), verifier_ids: Array(Int32))
  raise "Metal unavailable for self-draft baseline" unless ML::GGUF::Qwen35Metal.available?
  raise "n_draft must be positive" unless n_draft > 0
  raise "calib_count must leave a non-empty held-out span >= n_draft" unless calib_count + n_draft <= token_ids.size
  raise "layer_bases must not be empty" if layer_bases.empty?

  hp = weights.hparams
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  prefix_ids = token_ids[0, calib_count]
  input_ids = token_ids[calib_count, n_draft]
  max_seq = token_ids.size + n_draft + 8

  basis_size_bytes = (h_k * rank * s).to_i64 * sizeof(Float32)
  state_size_bytes = (h_v * s * rank).to_i64 * sizeof(Float32)
  full_state_size = h_v * s * s

  lowrank_set = Set(Int32).new(layer_bases.keys)
  shared_basis_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    buf = ML::MetalBuffer.new(basis_size_bytes)
    buf.write(flatten_basis_for_metal(bs, rank, h_k, s))
    shared_basis_bufs[il] = buf
  end

  build_lr_states = ->(state : ML::GGUF::Qwen35CPU::State) {
    bufs = {} of Int32 => ML::MetalBuffer
    layer_bases.each do |il, bs|
      ssm_buf = state.layers[il].ssm_state_buf
      buf = if ssm_buf
              ML::GGUF::Qwen35Metal.lowrank_project_state_buf(ssm_buf, shared_basis_bufs[il],
                h_k, h_v, s, rank, command_queue_name: "self_spec_gpu_pipeline_draft")
            else
              cpu_buf = ML::MetalBuffer.new(state_size_bytes)
              full_state = state.layers[il].ssm_state ||= Array(Float32).new(full_state_size, 0.0_f32)
              cpu_buf.write(project_full_state_to_lowrank(full_state, bs, rank, h_k, h_v, s))
              cpu_buf
            end
      bufs[il] = buf
    end
    bufs
  }

  warmup_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  warmup_lr = build_lr_states.call(warmup_state)
  ML::GGUF::Qwen35CPU.forward_self_draft_top1(weights, input_ids[0], calib_count, warmup_state,
    lowrank_set, warmup_lr, shared_basis_bufs, rank).not_nil!

  warmup_exact = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.forward_top1(weights, input_ids[0], calib_count, warmup_exact)

  warmup_verify = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, input_ids, calib_count, warmup_verify)

  self_draft_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  self_lr = build_lr_states.call(self_draft_state)
  self_draft_ids = [] of Int32
  t_self = Time.instant
  input_ids.each_with_index do |tok, i|
    out = ML::GGUF::Qwen35CPU.forward_self_draft_top1(weights, tok, calib_count + i, self_draft_state,
      lowrank_set, self_lr, shared_basis_bufs, rank).not_nil!
    self_draft_ids << out[0]
  end
  self_draft_ms = (Time.instant - t_self).total_milliseconds

  exact_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  exact_ids = [] of Int32
  t_exact = Time.instant
  input_ids.each_with_index do |tok, i|
    out = ML::GGUF::Qwen35CPU.forward_top1(weights, tok, calib_count + i, exact_state)
    exact_ids << out[0]
  end
  exact_ms = (Time.instant - t_exact).total_milliseconds

  verify_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_verify = Time.instant
  verify_results = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, input_ids, calib_count, verify_state)
  verifier_ms = (Time.instant - t_verify).total_milliseconds
  verifier_ids = verify_results.map { |r| r[0] }

  agreement = self_draft_ids.zip(exact_ids).count { |pair| pair[0] == pair[1] }
  wall_total = self_draft_ms + verifier_ms

  {
    steps:                   n_draft,
    self_draft_ms:           self_draft_ms,
    exact_ms:                exact_ms,
    verifier_ms:             verifier_ms,
    self_draft_per_token_ms: self_draft_ms / n_draft,
    exact_per_token_ms:      exact_ms / n_draft,
    verifier_per_token_ms:   verifier_ms / n_draft,
    self_spec_wall_ratio:    wall_total > 0.0 ? exact_ms / wall_total : 0.0,
    agreement:               agreement,
    self_draft_ids:          self_draft_ids,
    exact_ids:               exact_ids,
    verifier_ids:            verifier_ids,
  }
end

private def simulate_self_draft_gpu_chain_run(weights : ML::GGUF::Qwen35Weights,
                                              token_ids : Array(Int32),
                                              calib_count : Int32,
                                              n_draft : Int32,
                                              layer_bases : LayerBasisMap,
                                              rank : Int32,
                                              draft_updown_rank : Int32? = nil,
                                              ffn_updown_adapters : FFNUpDownAdapterMap? = nil,
                                              draft_updown_layer_indices : Set(Int32)? = nil,
                                              capture_top2 : Bool = false) : NamedTuple(steps: Int32, submit_ms: Float64, wait_ms: Float64, chain_ms: Float64, exact_ms: Float64, agreement: Int32, chain_ids: Array(Int32), chain_second_ids: Array(Int32), chain_top2_margins: Array(Float64), exact_ids: Array(Int32), updown_rank: Int32)
  raise "self-draft GPU chain requires at least one held-out token" unless n_draft > 0
  raise "self-draft GPU chain requires Metal" unless ML::GGUF::Qwen35Metal.available?
  hp = weights.hparams
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  prefix_ids = token_ids[0, calib_count]
  first_token = token_ids[calib_count]
  max_seq = token_ids.size + n_draft + 8

  basis_size_bytes = (h_k * rank * s).to_i64 * sizeof(Float32)
  state_size_bytes = (h_v * s * rank).to_i64 * sizeof(Float32)
  full_state_size = h_v * s * s

  lowrank_set = Set(Int32).new(layer_bases.keys)
  shared_basis_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    buf = ML::MetalBuffer.new(basis_size_bytes)
    buf.write(flatten_basis_for_metal(bs, rank, h_k, s))
    shared_basis_bufs[il] = buf
  end
  updown_x_mean_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_c_mean_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_coeff_w_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_down_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  actual_updown_rank = 0
  if requested_updown_rank = draft_updown_rank
    adapters = ffn_updown_adapters || raise "self-draft GPU chain pca-updown requires FFN up/down adapters"
    updown_layers = draft_updown_layer_indices || lowrank_set
    maps = build_updown_adapter_buffer_maps(adapters, updown_layers, requested_updown_rank, hp.n_embd)
    updown_x_mean_bufs = maps[:x_mean]
    updown_c_mean_bufs = maps[:c_mean]
    updown_coeff_w_bufs = maps[:coeff_w]
    updown_down_bufs = maps[:down]
    actual_updown_rank = maps[:rank]
  end

  build_lr_states = ->(state : ML::GGUF::Qwen35CPU::State) {
    bufs = {} of Int32 => ML::MetalBuffer
    layer_bases.each do |il, bs|
      buf = ML::MetalBuffer.new(state_size_bytes)
      ssm_buf = state.layers[il].ssm_state_buf
      full_state = if ssm_buf
                     ssm_buf.read(full_state_size)
                   else
                     state.layers[il].ssm_state ||= Array(Float32).new(full_state_size, 0.0_f32)
                   end
      buf.write(project_full_state_to_lowrank(full_state, bs, rank, h_k, h_v, s))
      bufs[il] = buf
    end
    bufs
  }

  exact_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  exact_ids = [] of Int32
  exact_tok = first_token
  t_exact = Time.instant
  n_draft.times do |i|
    out = ML::GGUF::Qwen35CPU.forward_top1(weights, exact_tok, calib_count + i, exact_state)
    exact_ids << out[0]
    exact_tok = out[0]
  end
  exact_ms = (Time.instant - t_exact).total_milliseconds

  chain_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  chain_lr = build_lr_states.call(chain_state)
  initial_token_buf = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
  initial_token_buf.contents.as(Pointer(UInt32)).value = first_token.to_u32
  token_buf = initial_token_buf
  submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  wba = WbaTrace.maybe("self_draft_gpu_chain")

  t_chain = Time.instant
  t_submit = Time.instant
  n_draft.times do |i|
    t0 = Time.instant
    sub = if capture_top2
            ML::GGUF::Qwen35CPU.forward_self_draft_top2_from_token_buf_async(weights, token_buf, 0, calib_count + i, chain_state,
              lowrank_set, chain_lr, shared_basis_bufs, rank,
              lowrank_updown_x_mean_bufs: updown_x_mean_bufs,
              lowrank_updown_c_mean_bufs: updown_c_mean_bufs,
              lowrank_updown_coeff_w_bufs: updown_coeff_w_bufs,
              lowrank_updown_down_bufs: updown_down_bufs,
              lowrank_updown_rank: actual_updown_rank,
              lowrank_updown_layer_indices: draft_updown_layer_indices,
              scratch_namespace: "self_draft_gpu_chain_#{i}").not_nil!
          else
            ML::GGUF::Qwen35CPU.forward_self_draft_top1_from_token_buf_async(weights, token_buf, 0, calib_count + i, chain_state,
              lowrank_set, chain_lr, shared_basis_bufs, rank,
              lowrank_updown_x_mean_bufs: updown_x_mean_bufs,
              lowrank_updown_c_mean_bufs: updown_c_mean_bufs,
              lowrank_updown_coeff_w_bufs: updown_coeff_w_bufs,
              lowrank_updown_down_bufs: updown_down_bufs,
              lowrank_updown_rank: actual_updown_rank,
              lowrank_updown_layer_indices: draft_updown_layer_indices,
              scratch_namespace: "self_draft_gpu_chain_#{i}").not_nil!
          end
    wba.try(&.mark("draft", "submit_#{i}", t0, Time.instant))
    submissions << sub
    token_buf = sub.top1_id_buf.not_nil!
  end
  submit_ms = (Time.instant - t_submit).total_milliseconds

  chain_ids = [] of Int32
  chain_second_ids = [] of Int32
  chain_top2_margins = [] of Float64
  t_wait = Time.instant
  submissions.each_with_index do |sub, i|
    t0 = Time.instant
    packed = ML::GGUF::Qwen35Metal.wait_forward_decode_wave(sub)
    wba.try(&.mark("draft", "wait_read_#{i}", t0, Time.instant))
    expected_packed_size = capture_top2 ? 4 : 2
    raise "GPU chain decode returned #{packed.size} values" unless packed.size == expected_packed_size
    chain_ids << packed[0].to_i32
    if capture_top2
      chain_second_ids << packed[2].to_i32
      chain_top2_margins << (packed[1] - packed[3]).to_f64
    end
  end
  wba.try(&.flush)
  wait_ms = (Time.instant - t_wait).total_milliseconds
  chain_ms = (Time.instant - t_chain).total_milliseconds
  agreement = chain_ids.zip(exact_ids).count { |pair| pair[0] == pair[1] }

  {
    steps:       n_draft,
    submit_ms:   submit_ms,
    wait_ms:     wait_ms,
    chain_ms:    chain_ms,
    exact_ms:    exact_ms,
    agreement:   agreement,
    chain_ids:   chain_ids,
    chain_second_ids: chain_second_ids,
    chain_top2_margins: chain_top2_margins,
    exact_ids:   exact_ids,
    updown_rank: actual_updown_rank,
  }
end

private def tuple_topk_rank(topk : Array({Int32, Float32}), target : Int32) : Int32
  topk.each_with_index do |(id, _), i|
    return i + 1 if id == target
  end
  0
end

private def pct_count(num : Int32, den : Int32) : Float64
  return 0.0 if den == 0
  100.0 * num / den
end

private def mtp_hidden_topk_for_fusion(weights : ML::GGUF::Qwen35Weights,
                                       mtp : ML::GGUF::Qwen35MTPWeights,
                                       prev_hidden : Array(Float32),
                                       token_id : Int32,
                                       pos : Int32,
                                       k : Int32)
  start = Time.instant
  next_hidden = ML::GGUF::Qwen35MTP.forward_one_hidden(weights, mtp, prev_hidden, token_id, pos, normalized: true)
  topk = if k == 1
           [ML::GGUF::Qwen35MTP.hidden_top1(weights, next_hidden)]
         else
           logits = ML::GGUF::Qwen35CPU.qmatvec_nobias(weights.output, next_hidden)
           ML::GGUF::Qwen35MTP.top_k(logits, k)
         end
  {hidden: next_hidden, topk: topk, ms: (Time.instant - start).total_milliseconds}
end

private def self_first_attempt_count(self_id : Int32, exact_id : Int32, mtp_topk : Array({Int32, Float32})) : Int32
  return 1 if self_id == exact_id

  mtp_rank = tuple_topk_rank(mtp_topk, exact_id)
  mtp_ids = mtp_topk.map(&.[0])
  union_size = mtp_ids.includes?(self_id) ? mtp_topk.size : mtp_topk.size + 1
  return union_size if mtp_rank == 0

  self_rank = tuple_topk_rank(mtp_topk, self_id)
  skipped_before_exact = self_rank > 0 && self_rank < mtp_rank ? 1 : 0
  1 + mtp_rank - skipped_before_exact
end

private def simulate_mtp_self_draft_fusion_run(weights : ML::GGUF::Qwen35Weights,
                                               mtp : ML::GGUF::Qwen35MTPWeights,
                                               token_ids : Array(Int32),
                                               calib_count : Int32,
                                               n_draft : Int32,
                                               layer_bases : LayerBasisMap,
                                               rank : Int32,
                                               mtp_topk : Int32,
                                               draft_updown_rank : Int32? = nil,
                                               ffn_updown_adapters : FFNUpDownAdapterMap? = nil,
                                               draft_updown_layer_indices : Set(Int32)? = nil)
  raise "MTP/self-draft fusion requires at least one held-out token" unless n_draft > 0
  raise "MTP/self-draft fusion requires positive MTP top-K" unless mtp_topk > 0

  chain = simulate_self_draft_gpu_chain_run(weights, token_ids, calib_count, n_draft, layer_bases, rank,
    draft_updown_rank, ffn_updown_adapters, draft_updown_layer_indices, capture_top2: true)
  hp = weights.hparams
  prefix_ids = token_ids[0, calib_count]
  first_token = token_ids[calib_count]
  max_seq = token_ids.size + n_draft + 8

  prefix_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(prefix_state, hp)
  prefix_hidden = ML::GGUF::Qwen35CPU.prefill_tokens_last_hidden(weights, prefix_ids, 0, prefix_state)

  exact_state = prefix_state.fork
  exact_hiddens = [] of Array(Float32)
  exact_ids = [] of Int32
  exact_tok = first_token
  n_draft.times do |i|
    hidden = ML::GGUF::Qwen35CPU.forward_hidden(weights, exact_tok, calib_count + i, exact_state)
    exact_id, _ = ML::GGUF::Qwen35CPU.hidden_top1(weights, hidden)
    exact_hiddens << hidden
    exact_ids << exact_id
    exact_tok = exact_id
  end
  raise "self-draft exact chain mismatch" unless exact_ids == chain[:exact_ids]

  rows = [] of MtpSelfDraftFusionRow
  mtp_prev_hidden = prefix_hidden
  mtp_prev_token = first_token
  mtp_pos = calib_count
  mtp_ms = 0.0
  self_hits = 0
  self_top2_hits = 0
  mtp_hits = 0
  mtp_k2_hits = 0
  union_hits = 0
  union_k2_hits = 0
  mtp_extra_over_self_top2 = 0
  mtp_extra_over_self_top2_topk = 0
  self_top2_extra_over_mtp_k2 = 0
  self_top2_extra_over_mtp_topk = 0
  both_top2_hits = 0
  union_topk_with_self_top2_hits = 0
  union_k2_size_total = 0
  agreement = 0
  agreement_hits = 0
  agreement_false = 0
  additive_hits = 0
  mtp_first_attempts_total = 0
  self_first_attempts_total = 0

  n_draft.times do |i|
    mtp_out = mtp_hidden_topk_for_fusion(weights, mtp, mtp_prev_hidden, mtp_prev_token, mtp_pos, mtp_topk)
    mtp_ms += mtp_out[:ms]
    mtp_top = mtp_out[:topk]
    mtp_ids = mtp_top.map(&.[0])
    exact_id = exact_ids[i]
    self_id = chain[:chain_ids][i]
    self_second_id = chain[:chain_second_ids][i]
    mtp_rank = tuple_topk_rank(mtp_top, exact_id)
    self_hit = self_id == exact_id
    self_top2_hit = self_hit || self_second_id == exact_id
    mtp_hit = mtp_rank > 0
    mtp_k2_count = Math.min(2, mtp_top.size)
    mtp_k2_ids = mtp_top[0, mtp_k2_count].map(&.[0])
    mtp_k2_hit = mtp_k2_ids.includes?(exact_id)
    union_hit = self_hit || mtp_hit
    union_topk_with_self_top2_hit = self_top2_hit || mtp_hit
    union_k2_ids = Set(Int32).new
    union_k2_ids << self_id
    union_k2_ids << self_second_id
    mtp_k2_ids.each { |id| union_k2_ids << id }
    union_k2_hit = union_k2_ids.includes?(exact_id)
    agrees = self_id == mtp_top[0][0]
    union_size = mtp_ids.includes?(self_id) ? mtp_top.size : mtp_top.size + 1
    union_k2_size = union_k2_ids.size
    mtp_first_attempts = mtp_hit ? mtp_rank : union_size
    self_first_attempts = self_first_attempt_count(self_id, exact_id, mtp_top)

    self_hits += 1 if self_hit
    self_top2_hits += 1 if self_top2_hit
    mtp_hits += 1 if mtp_hit
    mtp_k2_hits += 1 if mtp_k2_hit
    union_hits += 1 if union_hit
    union_k2_hits += 1 if union_k2_hit
    mtp_extra_over_self_top2 += 1 if mtp_k2_hit && !self_top2_hit
    mtp_extra_over_self_top2_topk += 1 if mtp_hit && !self_top2_hit
    self_top2_extra_over_mtp_k2 += 1 if self_top2_hit && !mtp_k2_hit
    self_top2_extra_over_mtp_topk += 1 if self_top2_hit && !mtp_hit
    both_top2_hits += 1 if self_top2_hit && mtp_k2_hit
    union_topk_with_self_top2_hits += 1 if union_topk_with_self_top2_hit
    union_k2_size_total += union_k2_size
    additive_hits += 1 if self_hit && !mtp_hit
    agreement += 1 if agrees
    agreement_hits += 1 if agrees && self_hit
    agreement_false += 1 if agrees && !self_hit
    mtp_first_attempts_total += mtp_first_attempts
    self_first_attempts_total += self_first_attempts
    rows << {
      index:               i.to_i32,
      exact:               exact_id,
      self_id:             self_id,
      self_second_id:      self_second_id,
      mtp_rank:            mtp_rank,
      self_hit:            self_hit,
      self_top2_hit:       self_top2_hit,
      mtp_hit:             mtp_hit,
      mtp_k2_hit:          mtp_k2_hit,
      union_hit:           union_hit,
      union_k2_hit:        union_k2_hit,
      agreement:           agrees,
      union_size:          union_size,
      union_k2_size:       union_k2_size,
      mtp_first_attempts:  mtp_first_attempts,
      self_first_attempts: self_first_attempts,
    }

    mtp_prev_hidden = exact_hiddens[i]
    mtp_prev_token = exact_id
    mtp_pos += 1
  end

  {
    steps:                     n_draft,
    rank:                      rank,
    mtp_topk:                  mtp_topk,
    self_hits:                 self_hits,
    self_top2_hits:            self_top2_hits,
    mtp_hits:                  mtp_hits,
    mtp_k2_hits:               mtp_k2_hits,
    union_hits:                union_hits,
    union_k2_hits:             union_k2_hits,
    union_topk_with_self_top2_hits: union_topk_with_self_top2_hits,
    mtp_extra_over_self_top2:  mtp_extra_over_self_top2,
    mtp_extra_over_self_top2_topk: mtp_extra_over_self_top2_topk,
    self_top2_extra_over_mtp_k2: self_top2_extra_over_mtp_k2,
    self_top2_extra_over_mtp_topk: self_top2_extra_over_mtp_topk,
    both_top2_hits:            both_top2_hits,
    union_k2_size_total:       union_k2_size_total,
    agreement:                 agreement,
    agreement_hits:            agreement_hits,
    agreement_false:           agreement_false,
    additive_hits:             additive_hits,
    mtp_first_attempts_total:  mtp_first_attempts_total,
    self_first_attempts_total: self_first_attempts_total,
    self_chain_ms:             chain[:chain_ms],
    self_exact_ms:             chain[:exact_ms],
    mtp_ms:                    mtp_ms,
    draft_updown_rank:         chain[:updown_rank],
    rows:                      rows,
    self_ids:                  chain[:chain_ids],
    self_second_ids:           chain[:chain_second_ids],
    self_top2_margins:         chain[:chain_top2_margins],
    exact_ids:                 exact_ids,
  }
end

private def simulate_self_draft_gpu_state_only_run(weights : ML::GGUF::Qwen35Weights,
                                                   token_ids : Array(Int32),
                                                   calib_count : Int32,
                                                   n_draft : Int32,
                                                   layer_bases : LayerBasisMap,
                                                   rank : Int32,
                                                   draft_updown_rank : Int32? = nil,
                                                   ffn_updown_adapters : FFNUpDownAdapterMap? = nil,
                                                   draft_updown_layer_indices : Set(Int32)? = nil) : NamedTuple(steps: Int32, project_ms: Float64, submit_ms: Float64, wait_ms: Float64, chain_ms: Float64, per_token_ms: Float64, updown_rank: Int32)
  raise "self-draft GPU state-only requires at least one held-out token" unless n_draft > 0
  raise "self-draft GPU state-only requires Metal" unless ML::GGUF::Qwen35Metal.available?
  hp = weights.hparams
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  prefix_ids = token_ids[0, calib_count]
  known_tokens = token_ids[calib_count, n_draft]
  max_seq = token_ids.size + n_draft + 8

  basis_size_bytes = (h_k * rank * s).to_i64 * sizeof(Float32)
  state_size_bytes = (h_v * s * rank).to_i64 * sizeof(Float32)
  full_state_size = h_v * s * s

  lowrank_set = Set(Int32).new(layer_bases.keys)
  shared_basis_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    buf = ML::MetalBuffer.new(basis_size_bytes)
    buf.write(flatten_basis_for_metal(bs, rank, h_k, s))
    shared_basis_bufs[il] = buf
  end
  updown_x_mean_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_c_mean_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_coeff_w_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_down_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  actual_updown_rank = 0
  if requested_updown_rank = draft_updown_rank
    adapters = ffn_updown_adapters || raise "self-draft GPU state-only pca-updown requires FFN up/down adapters"
    updown_layers = draft_updown_layer_indices || lowrank_set
    maps = build_updown_adapter_buffer_maps(adapters, updown_layers, requested_updown_rank, hp.n_embd)
    updown_x_mean_bufs = maps[:x_mean]
    updown_c_mean_bufs = maps[:c_mean]
    updown_coeff_w_bufs = maps[:coeff_w]
    updown_down_bufs = maps[:down]
    actual_updown_rank = maps[:rank]
  end

  state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  t_project = Time.instant
  lr_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    ssm_buf = state.layers[il].ssm_state_buf
    buf = if ssm_buf
            ML::GGUF::Qwen35Metal.lowrank_project_state_buf(ssm_buf, shared_basis_bufs[il],
              h_k, h_v, s, rank, command_queue_name: "self_draft_gpu_state_only")
          else
            cpu_buf = ML::MetalBuffer.new(state_size_bytes)
            full_state = state.layers[il].ssm_state ||= Array(Float32).new(full_state_size, 0.0_f32)
            cpu_buf.write(project_full_state_to_lowrank(full_state, bs, rank, h_k, h_v, s))
            cpu_buf
          end
    lr_bufs[il] = buf
  end
  project_ms = (Time.instant - t_project).total_milliseconds

  token_buf = ML::MetalBuffer.new(n_draft.to_i64 * sizeof(UInt32))
  token_ptr = token_buf.contents.as(Pointer(UInt32))
  known_tokens.each_with_index do |tok, i|
    token_ptr[i] = tok.to_u32
  end

  submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  wba = WbaTrace.maybe("self_draft_gpu_state_only")
  cmd = ML::GGUF::Qwen35Metal.decode_wave_command_buffer("self_draft_gpu_state_only")

  t_chain = Time.instant
  t_submit = Time.instant
  n_draft.times do |i|
    t0 = Time.instant
    sub = ML::GGUF::Qwen35CPU.forward_self_draft_state_from_token_buf_async(weights, token_buf, i, calib_count + i, state,
      lowrank_set, lr_bufs, shared_basis_bufs, rank,
      lowrank_updown_x_mean_bufs: updown_x_mean_bufs,
      lowrank_updown_c_mean_bufs: updown_c_mean_bufs,
      lowrank_updown_coeff_w_bufs: updown_coeff_w_bufs,
      lowrank_updown_down_bufs: updown_down_bufs,
      lowrank_updown_rank: actual_updown_rank,
      lowrank_updown_layer_indices: draft_updown_layer_indices,
      scratch_namespace: "self_draft_gpu_state_only_#{i}",
      command_queue_name: "self_draft_gpu_state_only",
      append_command_buffer: cmd).not_nil!
    wba.try(&.mark("draft", "submit_state_only_#{i}", t0, Time.instant))
    submissions << sub
  end
  cmd.commit
  submit_ms = (Time.instant - t_submit).total_milliseconds

  t_wait = Time.instant
  cmd.wait
  submissions.each do |sub|
    sub.pending_cmds.each(&.wait)
  end
  wait_ms = (Time.instant - t_wait).total_milliseconds
  wba.try(&.mark("draft", "wait_state_only_block", t_wait, Time.instant))
  wba.try(&.flush)
  chain_ms = (Time.instant - t_chain).total_milliseconds

  {
    steps:        n_draft,
    project_ms:   project_ms,
    submit_ms:    submit_ms,
    wait_ms:      wait_ms,
    chain_ms:     chain_ms,
    per_token_ms: chain_ms / n_draft,
    updown_rank:  actual_updown_rank,
  }
end

private def simulate_self_draft_gpu_chain_overlap_run(weights : ML::GGUF::Qwen35Weights,
                                                      token_ids : Array(Int32),
                                                      calib_count : Int32,
                                                      n_draft : Int32,
                                                      layer_bases : LayerBasisMap,
                                                      rank : Int32) : NamedTuple(steps: Int32, draft_alone_ms: Float64, verifier_ms: Float64, overlap_ms: Float64, draft_submit_ms: Float64, draft_wait_ms: Float64, hidden_ms: Float64, speedup: Float64, agreement: Int32, draft_ids: Array(Int32), exact_ids: Array(Int32), verifier_ids: Array(Int32))
  solo = simulate_self_draft_gpu_chain_run(weights, token_ids, calib_count, n_draft, layer_bases, rank)
  raise "self-draft GPU chain overlap requires Metal" unless ML::GGUF::Qwen35Metal.available?
  hp = weights.hparams
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  prefix_ids = token_ids[0, calib_count]
  candidates = token_ids[calib_count, n_draft]
  first_token = candidates[0]
  max_seq = token_ids.size + n_draft + 8

  basis_size_bytes = (h_k * rank * s).to_i64 * sizeof(Float32)
  state_size_bytes = (h_v * s * rank).to_i64 * sizeof(Float32)
  full_state_size = h_v * s * s

  lowrank_set = Set(Int32).new(layer_bases.keys)
  shared_basis_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    buf = ML::MetalBuffer.new(basis_size_bytes)
    buf.write(flatten_basis_for_metal(bs, rank, h_k, s))
    shared_basis_bufs[il] = buf
  end

  build_lr_states = ->(state : ML::GGUF::Qwen35CPU::State) {
    bufs = {} of Int32 => ML::MetalBuffer
    layer_bases.each do |il, bs|
      buf = ML::MetalBuffer.new(state_size_bytes)
      ssm_buf = state.layers[il].ssm_state_buf
      full_state = if ssm_buf
                     ssm_buf.read(full_state_size)
                   else
                     state.layers[il].ssm_state ||= Array(Float32).new(full_state_size, 0.0_f32)
                   end
      buf.write(project_full_state_to_lowrank(full_state, bs, rank, h_k, h_v, s))
      bufs[il] = buf
    end
    bufs
  }

  draft_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  draft_lr = build_lr_states.call(draft_state)
  verifier_state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  initial_token_buf = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
  initial_token_buf.contents.as(Pointer(UInt32)).value = first_token.to_u32
  token_buf = initial_token_buf
  submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
  wba = WbaTrace.maybe("self_draft_gpu_chain_overlap")

  t_overlap = Time.instant
  t_submit = Time.instant
  n_draft.times do |i|
    t0 = Time.instant
    sub = ML::GGUF::Qwen35CPU.forward_self_draft_top1_from_token_buf_async(weights, token_buf, 0, calib_count + i, draft_state,
      lowrank_set, draft_lr, shared_basis_bufs, rank,
      scratch_namespace: "self_draft_gpu_chain_overlap_#{i}",
      command_queue_name: "self_draft_gpu_chain_overlap").not_nil!
    wba.try(&.mark("draft", "submit_#{i}", t0, Time.instant))
    submissions << sub
    token_buf = sub.top1_id_buf.not_nil!
  end
  draft_submit_ms = (Time.instant - t_submit).total_milliseconds

  t_verify = Time.instant
  verifier = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, candidates, calib_count, verifier_state)
  verifier_ms = (Time.instant - t_verify).total_milliseconds
  wba.try(&.mark("verifier", "chunk_major", t_verify, Time.instant))

  draft_ids = [] of Int32
  t_wait = Time.instant
  submissions.each_with_index do |sub, i|
    t0 = Time.instant
    packed = ML::GGUF::Qwen35Metal.wait_forward_decode_wave(sub)
    wba.try(&.mark("draft", "wait_read_#{i}", t0, Time.instant))
    raise "GPU chain overlap decode returned #{packed.size} values" unless packed.size == 2
    draft_ids << packed[0].to_i32
  end
  draft_wait_ms = (Time.instant - t_wait).total_milliseconds
  overlap_ms = (Time.instant - t_overlap).total_milliseconds
  wba.try(&.flush)

  serial_ms = solo[:chain_ms] + verifier_ms
  hidden_ms = serial_ms - overlap_ms
  {
    steps:           n_draft,
    draft_alone_ms:  solo[:chain_ms],
    verifier_ms:     verifier_ms,
    overlap_ms:      overlap_ms,
    draft_submit_ms: draft_submit_ms,
    draft_wait_ms:   draft_wait_ms,
    hidden_ms:       hidden_ms,
    speedup:         overlap_ms > 0.0 ? serial_ms / overlap_ms : 0.0,
    agreement:       draft_ids.zip(solo[:exact_ids]).count { |pair| pair[0] == pair[1] },
    draft_ids:       draft_ids,
    exact_ids:       solo[:exact_ids],
    verifier_ids:    verifier.map { |r| r[0] },
  }
end

private def simulate_self_spec_gpu_pipeline_exact_fallback_run(weights : ML::GGUF::Qwen35Weights,
                                                               prompt_ids : Array(Int32),
                                                               gen_tokens : Int32)
  raise "exact fallback requires non-empty prompt" if prompt_ids.empty?
  raise "exact fallback gen_tokens must be positive" unless gen_tokens > 0

  max_seq = prompt_ids.size + gen_tokens + 8
  prefix_ids = prompt_ids[0, prompt_ids.size - 1]
  state = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  last_token = prompt_ids[-1]
  pos_last = prompt_ids.size - 1
  ids = [] of Int32
  t0 = Time.instant
  gen_tokens.times do
    id = ML::GGUF::Qwen35CPU.forward_top1(weights, last_token, pos_last, state)[0]
    ids << id
    last_token = id
    pos_last += 1
  end
  ms = (Time.instant - t0).total_milliseconds
  {
    plain_exact_ms: ms,
    serial_ms:      ms,
    overlap_ms:     ms,
    speedup:        1.0,
    plain_speedup:  1.0,
    parity:         true,
    exact_ids:      ids,
    emitted_ids:    ids,
  }
end

private def self_spec_pipeline_exact_fallback_fields(exact, gen_tokens : Int32) : String
  " gen_tokens=#{gen_tokens} chunks=0 draft_updown_chunks=0 rejections=0 accepted_draft_tokens=0 proposed_tokens=0 accept_rate=0.0% parity=#{exact[:parity]} gamma_history=exact draft_seed_ms=0.0 draft_next_ms=0.0 verifier_ms=0.0 draft_wait_ms=0.0 backup_ms=0.0 rebuild_ms=0.0 controller_ms=0.0 replay_ms=0.0 plain_exact_ms=#{exact[:plain_exact_ms].round(3)} serial_ms=#{exact[:serial_ms].round(3)} overlap_ms=#{exact[:overlap_ms].round(3)} hidden_ms=0.0 speedup=#{exact[:speedup].round(4)}x plain_speedup=#{exact[:plain_speedup].round(4)}x exact_ids=#{exact[:exact_ids].join(',')} emitted_ids=#{exact[:emitted_ids].join(',')}"
end

private def trace_self_spec_router_token(chunk : Int32,
                                         index : Int32,
                                         generated_offset : Int32,
                                         chunk_size : Int32,
                                         verifier_size : Int32,
                                         draft_margin : Float64?,
                                         proposal_margin_min : Float64,
                                         top1_hit : Bool,
                                         top2_hit : Bool,
                                         second_id_available : Bool,
                                         reject : Bool,
                                         branch_guard_index : Int32?,
                                         margin_guard_index : Int32?,
                                         risk_offramp : Bool,
                                         next_pre_submitted : Bool,
                                         draft_updown : Bool,
                                         rejections_before : Int32,
                                         accepted_before : Int32)
  return unless io = ProbeRuntime.self_spec_router_trace_io

  guard_role = if bgi = branch_guard_index
                 if index < bgi
                   "branch_prefix"
                 elsif index == bgi
                   "branch_guard"
                 else
                   "branch_suffix"
                 end
               elsif gi = margin_guard_index
                 if index < gi
                   "margin_prefix"
                 elsif index == gi
                   "margin_guard"
                 else
                   "margin_suffix"
                 end
               else
                 "none"
               end

  JSON.build(io) do |json|
    json.object do
      json.field "label", ProbeRuntime.self_spec_router_trace_label
      json.field "chunk", chunk
      json.field "index", index
      json.field "generated_offset", generated_offset
      json.field "chunk_size", chunk_size
      json.field "verifier_size", verifier_size
      json.field "final_tail_skip", index >= verifier_size
      json.field "first_in_chunk", index == 0
      json.field "last_in_chunk", index == chunk_size - 1
      if margin = draft_margin
        json.field "draft_margin", margin
      else
        json.field "draft_margin", nil
      end
      if proposal_margin_min == Float64::INFINITY
        json.field "proposal_margin_min", nil
      else
        json.field "proposal_margin_min", proposal_margin_min
      end
      json.field "top1_hit", top1_hit
      json.field "top2_hit", top2_hit
      json.field "second_id_available", second_id_available
      json.field "reject", reject
      json.field "branch_guard_index", branch_guard_index || -1
      json.field "margin_guard_index", margin_guard_index || -1
      json.field "guard_role", guard_role
      json.field "risk_offramp", risk_offramp
      json.field "next_pre_submitted", next_pre_submitted
      json.field "draft_updown", draft_updown
      json.field "rejections_before", rejections_before
      json.field "accepted_before", accepted_before
    end
  end
  io << '\n'
end

private def dump_self_spec_gpu_pipeline_cycles(path : String,
                                               prompt_name : String,
                                               prompt_text : String,
                                               mode : String,
                                               layers : Array(Int32),
                                               rank : Int32,
                                               gamma_label : String,
                                               pipe) : Nil
  Dir.mkdir_p(File.dirname(path))
  prompt_hash = fnv1a64_hex(prompt_text.to_slice)
  prompt_category = probe_prompt_category(prompt_name)
  accepted_history = pipe[:accept_history]
  proposed_history = pipe[:gamma_history]
  reject_history = pipe[:reject_index_history]
  generated_total = pipe[:emitted_ids].size
  proposed_total = Math.max(pipe[:proposed_tokens], 1)
  plain_ms_per_token = generated_total > 0 ? pipe[:plain_exact_ms] / generated_total : 0.0
  emitted_before = 0
  assigned_wall_ms = 0.0

  File.open(path, "a") do |io|
    proposed_history.each_with_index do |proposed_count, i|
      accepted_count = accepted_history[i]? || 0
      reject_index = reject_history[i]? || -1
      generated_count = accepted_count + (reject_index >= 0 ? 1 : 0)
      generated_count = accepted_count if reject_index < 0
      generated_count = Math.min(generated_count, generated_total - emitted_before)
      generated_count = Math.max(generated_count, 0)
      gen_share = generated_total > 0 ? generated_count.to_f64 / generated_total : 0.0
      prop_share = proposed_count.to_f64 / proposed_total
      wall_ms = pipe[:overlap_ms] * gen_share
      assigned_wall_ms += wall_ms
      candidate_fingerprint = "#{prompt_hash}:#{mode}:#{gamma_label}:#{i}:#{proposed_count}:#{accepted_count}:#{reject_index}"

      JSON.build(io) do |json|
        json.object do
          json.field "prompt_hash", prompt_hash
          json.field "target_model", "qwen35"
          json.field "draft_model", "qwen35_self_lowrank"
          json.field "kind", "self_lowrank"
          json.field "policy", mode
          json.field "verify_mode", "exact_self_verify"
          json.field "prompt_category", prompt_category
          json.field "prompt_name", prompt_name
          json.field "position", emitted_before
          json.field "generated_before", emitted_before
          json.field "generated_count", generated_count
          json.field "gamma", proposed_count
          json.field "gamma_label", gamma_label
          json.field "proposed_count", proposed_count
          json.field "accepted_count", accepted_count
          json.field "reject_index", reject_index
          json.field "ngram_match_len", 0
          json.field "ngram_min", 0
          json.field "ngram_max", 0
          json.field "ngram_recursive", false
          json.field "ngram_disabled_before", false
          json.field "ngram_disabled_after", false
          json.field "candidate_hash", fnv1a64_hex(candidate_fingerprint.to_slice)
          json.field "proposal_ms", 0.0
          json.field "accept_scan_ms", pipe[:controller_ms] * gen_share
          json.field "commit_ms", 0.0
          json.field "target_replay_ms", pipe[:replay_ms] * gen_share
          json.field "draft_ms", (pipe[:draft_seed_ms] + pipe[:draft_next_ms] + pipe[:draft_wait_ms]) * prop_share
          json.field "target_verify_ms", pipe[:verifier_ms] * gen_share
          json.field "target_backup_ms", pipe[:backup_ms] * gen_share
          json.field "draft_backup_ms", 0.0
          json.field "draft_resync_ms", reject_index >= 0 ? pipe[:draft_resync_ms] * gen_share : 0.0
          json.field "wall_ms", wall_ms
          json.field "expected_gain_ms", generated_count * plain_ms_per_token - wall_ms
          json.field "plain_exact_ms", pipe[:plain_exact_ms]
          json.field "serial_ms", pipe[:serial_ms]
          json.field "plain_speedup", pipe[:plain_speedup]
          json.field "parity", pipe[:parity]
          json.field "rank", rank
          json.field "layers", layers.join(",")
          json.field "atlas_scope", "chunk"
        end
      end
      io << '\n'
      emitted_before += generated_count
    end
    if emitted_before < generated_total
      generated_count = generated_total - emitted_before
      wall_ms = [pipe[:overlap_ms] - assigned_wall_ms, 0.0].max
      candidate_fingerprint = "#{prompt_hash}:#{mode}:#{gamma_label}:suffix:#{emitted_before}:#{generated_count}"
      JSON.build(io) do |json|
        json.object do
          json.field "prompt_hash", prompt_hash
          json.field "target_model", "qwen35"
          json.field "draft_model", "qwen35_self_lowrank"
          json.field "kind", "self_lowrank_exact_suffix"
          json.field "policy", mode
          json.field "verify_mode", "exact_self_verify"
          json.field "prompt_category", prompt_category
          json.field "prompt_name", prompt_name
          json.field "position", emitted_before
          json.field "generated_before", emitted_before
          json.field "generated_count", generated_count
          json.field "gamma", 0
          json.field "gamma_label", gamma_label
          json.field "proposed_count", 0
          json.field "accepted_count", generated_count
          json.field "reject_index", -1
          json.field "ngram_match_len", 0
          json.field "ngram_min", 0
          json.field "ngram_max", 0
          json.field "ngram_recursive", false
          json.field "ngram_disabled_before", false
          json.field "ngram_disabled_after", false
          json.field "candidate_hash", fnv1a64_hex(candidate_fingerprint.to_slice)
          json.field "proposal_ms", 0.0
          json.field "accept_scan_ms", 0.0
          json.field "commit_ms", 0.0
          json.field "target_replay_ms", 0.0
          json.field "draft_ms", 0.0
          json.field "target_verify_ms", wall_ms
          json.field "target_backup_ms", 0.0
          json.field "draft_backup_ms", 0.0
          json.field "draft_resync_ms", 0.0
          json.field "wall_ms", wall_ms
          json.field "expected_gain_ms", generated_count * plain_ms_per_token - wall_ms
          json.field "plain_exact_ms", pipe[:plain_exact_ms]
          json.field "serial_ms", pipe[:serial_ms]
          json.field "plain_speedup", pipe[:plain_speedup]
          json.field "parity", pipe[:parity]
          json.field "rank", rank
          json.field "layers", layers.join(",")
          json.field "atlas_scope", "exact_suffix"
        end
      end
      io << '\n'
    end
  end
end

private def simulate_self_spec_gpu_pipeline_run(weights : ML::GGUF::Qwen35Weights,
                                                prompt_ids : Array(Int32),
                                                gen_tokens : Int32,
                                                gamma : Int32,
                                                layer_bases : LayerBasisMap,
                                                rank : Int32,
                                                use_verifier_backup : Bool = true,
                                                draft_block_tokens : Int32? = nil,
                                                draft_no_ffn : Bool = false,
                                                draft_skip_recurrent_ffn : Bool = false,
                                                gamma_schedule : Array(Int32)? = nil,
                                                draft_updown_rank : Int32? = nil,
                                                ffn_updown_adapters : FFNUpDownAdapterMap? = nil,
                                                draft_updown_q8_metal : Bool = false,
                                                draft_updown_fallback_on_reject : Bool = false,
                                                draft_updown_after_full_accepts : Int32 = 0,
                                                draft_updown_min_margin : Float64? = nil,
                                                draft_updown_max_chunks : Int32? = nil,
                                                draft_updown_after_rejects : Int32 = 0,
                                                draft_updown_refresh_on_accept : Bool = false,
                                                draft_updown_agreement_gate : Bool = false,
                                                draft_updown_agreement_steps : Int32 = 1,
                                                draft_updown_agreement_margin_thresholds : Array(Float64) = [] of Float64,
                                                live_state_backup : Bool = true,
                                                draft_no_ffn_layer_indices : Set(Int32)? = nil,
                                                draft_updown_layer_indices : Set(Int32)? = nil,
                                                tree2_first : Bool = false,
                                                tree2_anywhere : Bool = false,
                                                tree2_staged_tokens : Int32 = 0,
                                                tree2_margin_guard : Float64? = nil,
                                                tree2_branch_guard : Float64? = nil,
                                                risk_offramp_margin : Float64? = nil,
                                                mtp_k2_on_reject : ML::GGUF::Qwen35MTPWeights? = nil,
                                                reject_offramp_after : Int32 = 0) : NamedTuple(chunks: Int32, rejections: Int32, accepted_draft_tokens: Int32, proposed_tokens: Int32, draft_updown_chunks: Int32, draft_noffn_chunks: Int32, draft_updown_agreement_checks: Int32, draft_updown_agreement_passes: Int32, draft_updown_agreement_top1: Int32, draft_updown_agreement_top2: Int32, draft_updown_agreement_fails: Int32, draft_updown_agreement_probe_ms: Float64, draft_updown_agreement_margin_min_avg: Float64, draft_updown_agreement_margin_pass_avg: Float64, draft_updown_agreement_margin_fail_avg: Float64, draft_updown_agreement_margin_sweep: String, tree2_first_checks: Int32, tree2_first_rescues: Int32, tree2_first_misses: Int32, tree2_first_early_exits: Int32, tree2_anywhere_checks: Int32, tree2_anywhere_rescues: Int32, tree2_anywhere_misses: Int32, tree2_anywhere_early_exits: Int32, tree2_staged_checks: Int32, tree2_staged_rescues: Int32, tree2_staged_misses: Int32, tree2_staged_early_exits: Int32, tree2_staged_stages: Int32, tree2_margin_checks: Int32, tree2_margin_avg: Float64, tree2_margin_min: Float64, tree2_reject_margin_checks: Int32, tree2_reject_margin_avg: Float64, tree2_reject_margin_min: Float64, tree2_margin_guard_threshold: Float64, tree2_margin_guard_hits: Int32, tree2_margin_guard_tokens: Int32, tree2_margin_guard_rejects: Int32, tree2_margin_guard_passes: Int32, tree2_branch_guard_threshold: Float64, tree2_branch_guard_hits: Int32, tree2_branch_guard_tokens: Int32, tree2_branch_guard_rejects: Int32, tree2_branch_guard_rescues: Int32, tree2_branch_guard_misses: Int32, tree2_branch_guard_passes: Int32, tree2_branch_guard_prefix_rejects: Int32, tree2_branch_guard_replayless_resyncs: Int32, tree2_branch_guard_snapshot_copies: Int32, tree2_branch_guard_snapshot_ms: Float64, tree2_branch_guard_snapshot_restore_ms: Float64, tree2_branch_guard_snapshot_resync_base_ms: Float64, tree2_branch_guard_suffix_replays: Int32, tree2_branch_guard_suffix_replay_tokens: Int32, tree2_branch_guard_suffix_replay_ms: Float64, tree2_branch_guard_prefix_verify_ms: Float64, tree2_branch_guard_prefix_verify_tokens: Int32, tree2_branch_guard_token_verify_ms: Float64, tree2_branch_guard_token_verify_tokens: Int32, tree2_branch_guard_suffix_verify_ms: Float64, tree2_branch_guard_suffix_verify_tokens: Int32, tree2_branch_guard_snapshot_suffix_verify_ms: Float64, tree2_branch_guard_snapshot_suffix_verify_tokens: Int32, tree2_branch_guard_no_snapshot_suffix_verify_ms: Float64, tree2_branch_guard_no_snapshot_suffix_verify_tokens: Int32, risk_offramp_threshold: Float64, risk_offramp_hits: Int32, risk_offramp_delayed_blocks: Int32, risk_offramp_delayed_tokens: Int32, mtp_k2_reject_checks: Int32, mtp_k2_reject_rescues: Int32, mtp_k2_reject_misses: Int32, mtp_k2_reject_ms: Float64, reject_offramp_after: Int32, reject_offramp_hits: Int32, reject_offramp_tokens: Int32, reject_offramp_ms: Float64, draft_seed_ms: Float64, draft_next_ms: Float64, verifier_ms: Float64, draft_wait_ms: Float64, backup_ms: Float64, rebuild_ms: Float64, controller_ms: Float64, plain_exact_ms: Float64, serial_ms: Float64, overlap_ms: Float64, replay_ms: Float64, hidden_ms: Float64, speedup: Float64, plain_speedup: Float64, parity: Bool, gamma_history: Array(Int32), accept_history: Array(Int32), reject_index_history: Array(Int32), exact_ids: Array(Int32), emitted_ids: Array(Int32), draft_steps: Int32, draft_blocks: Int32, draft_fork_ms: Float64, draft_token_buf_ms: Float64, draft_lr_project_ms: Float64, draft_submit_ms: Float64, draft_commit_ms: Float64, draft_wait_block_ms: Float64, draft_read_ids_ms: Float64, draft_resync_ms: Float64, draft_resyncs: Int32, draft_wasted_tail_tokens: Int32, draft_wasted_next_tokens: Int32, verifier_initial_ms: Float64, verifier_prefill_ms: Float64, verifier_chunks: Int32, verifier_tokens: Int32, verifier_tail_skip_tokens: Int32)
  raise "GPU pipeline requires Metal" unless ML::GGUF::Qwen35Metal.available?
  raise "GPU pipeline gamma must be positive" unless gamma > 0
  raise "GPU pipeline gen_tokens must be positive" unless gen_tokens > 0
  raise "GPU pipeline pca-updown warmup must be non-negative" if draft_updown_after_full_accepts < 0
  raise "GPU pipeline pca-updown min-margin gate must be non-negative" if (guard = draft_updown_min_margin) && guard < 0.0
  raise "GPU pipeline pca-updown max chunks must be non-negative" if !draft_updown_max_chunks.nil? && draft_updown_max_chunks.not_nil! < 0
  raise "GPU pipeline pca-updown after-rejects must be non-negative" if draft_updown_after_rejects < 0
  raise "GPU pipeline pca-updown agreement steps must be positive" if draft_updown_agreement_steps <= 0
  raise "GPU pipeline tree2 staged tokens must be non-negative" if tree2_staged_tokens < 0
  raise "GPU pipeline tree2 margin guard must be non-negative" if (guard = tree2_margin_guard) && guard < 0.0
  raise "GPU pipeline tree2 branch guard must be non-negative" if (guard = tree2_branch_guard) && guard < 0.0
  raise "GPU pipeline risk offramp margin must be non-negative" if (guard = risk_offramp_margin) && guard < 0.0
  raise "GPU pipeline reject offramp threshold must be non-negative" if reject_offramp_after < 0
  raise "GPU pipeline risk offramp currently cannot combine with tree2_anywhere/tree2_staged" if risk_offramp_margin && (tree2_anywhere || tree2_staged_tokens > 0)
  if reject_offramp_after > 0 && (tree2_anywhere || tree2_staged_tokens > 0 || !tree2_margin_guard.nil? || !tree2_branch_guard.nil? || !risk_offramp_margin.nil? || mtp_k2_on_reject)
    raise "GPU pipeline reject offramp currently supports only the plain route and tree2-first, not tree2-anywhere/staged/risk/MTP routes"
  end
  raise "GPU pipeline tree2 branch guard currently cannot combine with tree2_anywhere/tree2_staged/tree2_margin_guard" if tree2_branch_guard && (tree2_anywhere || tree2_staged_tokens > 0 || tree2_margin_guard)
  raise "GPU pipeline tree2 branch guard requires verifier backup" if tree2_branch_guard && !use_verifier_backup
  branch_guard_snapshot_enabled = ProbeRuntime.self_spec_branch_guard_snapshot
  raise "GPU pipeline tree2 branch guard snapshot requires --simulate-self-spec-gpu-pipeline-tree2-branch-guard" if branch_guard_snapshot_enabled && tree2_branch_guard.nil?
  branch_guard_snapshot_min_prefix = ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix
  branch_guard_snapshot_suffix_threshold = ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold
  branch_guard_snapshot_suffix_min_threshold = ProbeRuntime.self_spec_branch_guard_snapshot_suffix_min_threshold
  branch_guard_snapshot_prefix_suffix_thresholds = ProbeRuntime.self_spec_branch_guard_snapshot_prefix_suffix_thresholds
  branch_guard_until_reject = ProbeRuntime.self_spec_branch_guard_until_reject
  branch_guard_overlap_next = ProbeRuntime.self_spec_branch_guard_overlap_next
  branch_guard_snapshot_only_split = ProbeRuntime.self_spec_branch_guard_snapshot_only_split
  branch_guard_single_pass_checkpoint = ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint
  branch_guard_no_snapshot_threshold = ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold
  raise "GPU pipeline tree2 branch guard until-reject requires --simulate-self-spec-gpu-pipeline-tree2-branch-guard" if branch_guard_until_reject && tree2_branch_guard.nil?
  raise "GPU pipeline tree2 branch guard overlap-next requires --simulate-self-spec-gpu-pipeline-tree2-branch-guard" if branch_guard_overlap_next && tree2_branch_guard.nil?
  raise "GPU pipeline tree2 branch guard snapshot-only split requires snapshot mode" if branch_guard_snapshot_only_split && !branch_guard_snapshot_enabled
  raise "GPU pipeline tree2 branch guard single-pass checkpoint requires snapshot mode" if branch_guard_single_pass_checkpoint && !branch_guard_snapshot_enabled
  mtp_k2_on_reject_enabled = !mtp_k2_on_reject.nil?
  if mtp_k2_on_reject_enabled && (tree2_first || tree2_anywhere || tree2_staged_tokens > 0 || !tree2_margin_guard.nil? || !tree2_branch_guard.nil? || !risk_offramp_margin.nil?)
    raise "GPU pipeline MTP K2 reject diagnostic currently cannot combine with tree2/risk routes"
  end
  raise "GPU pipeline MTP K2 reject diagnostic requires verifier backup" if mtp_k2_on_reject_enabled && !use_verifier_backup
  raise "GPU pipeline requires non-empty prompt" if prompt_ids.empty?
  schedule = gamma_schedule && !gamma_schedule.not_nil!.empty? ? gamma_schedule.not_nil! : [gamma]
  raise "GPU pipeline schedule values must be positive" if schedule.any? { |v| v <= 0 }
  exact_refresh_interval = ProbeRuntime.gpu_draft_exact_refresh_interval
  exact_refresh_offsets = ProbeRuntime.gpu_draft_exact_refresh_offsets.dup
  exact_refresh_layer_offsets = {} of Int32 => Set(Int32)
  exact_refresh_prefix = ProbeRuntime.gpu_draft_exact_refresh_prefix
  draft_refresh_on_accept = ProbeRuntime.self_spec_draft_refresh_on_accept
  draft_no_ffn_fallback_on_reject = ProbeRuntime.self_spec_draft_no_ffn_fallback_on_reject
  draft_no_ffn_after_full_accepts = ProbeRuntime.self_spec_draft_no_ffn_after_full_accepts
  draft_no_ffn_min_margin = ProbeRuntime.self_spec_draft_no_ffn_min_margin
  draft_no_ffn_max_chunks = ProbeRuntime.self_spec_draft_no_ffn_max_chunks
  raise "GPU pipeline no-FFN after-full-accepts must be non-negative" if draft_no_ffn_after_full_accepts < 0
  raise "GPU pipeline no-FFN max chunks must be non-negative" if draft_no_ffn_max_chunks && draft_no_ffn_max_chunks.not_nil! < 0
  draft_no_ffn_candidate = draft_no_ffn || !draft_no_ffn_layer_indices.nil?
  draft_no_ffn_dynamic_gate = draft_no_ffn_candidate && (draft_no_ffn_after_full_accepts > 0 || !draft_no_ffn_min_margin.nil?)
  if (draft_no_ffn_dynamic_gate || !draft_no_ffn_max_chunks.nil?) && !draft_no_ffn_candidate
    raise "GPU pipeline no-FFN chunk gate requires --simulate-self-spec-gpu-pipeline-draft-no-ffn or --simulate-self-spec-gpu-pipeline-draft-no-ffn-layers"
  end
  active_draft_no_ffn = draft_no_ffn_candidate && !draft_no_ffn_dynamic_gate
  active_draft_no_ffn_layer_indices = active_draft_no_ffn ? draft_no_ffn_layer_indices : nil
  max_gamma = schedule.max
  router_trace_enabled = !ProbeRuntime.self_spec_router_trace_io.nil?
  tree2_enabled = tree2_first || tree2_anywhere || tree2_staged_tokens > 0 || !tree2_margin_guard.nil? || !tree2_branch_guard.nil? || !risk_offramp_margin.nil? || !draft_updown_min_margin.nil? || !ProbeRuntime.self_spec_draft_updown_first_margin_threshold.nil? || !draft_no_ffn_min_margin.nil? || draft_updown_agreement_gate || mtp_k2_on_reject_enabled || router_trace_enabled

  hp = weights.hparams
  copy_verifier_state = ->(dst : ML::GGUF::Qwen35CPU::State, src : ML::GGUF::Qwen35CPU::State, used_tokens : Int32) {
    if live_state_backup
      ML::GGUF::Qwen35CPU.copy_state_metal_used!(dst, src, hp, used_tokens: used_tokens)
    else
      dst.copy_from!(src)
    end
  }
  copy_verifier_recurrent_state = ->(dst : ML::GGUF::Qwen35CPU::State, src : ML::GGUF::Qwen35CPU::State, used_tokens : Int32) {
    if live_state_backup
      # For branch-snapshot restore, KV prefix rows before `used_tokens` are
      # still valid in the verifier state and suffix rows are replay-overwritten.
      # Only recurrent state has to rewind exactly to the guard boundary.
      ML::GGUF::Qwen35CPU.copy_state_metal_used!(dst, src, hp, used_tokens: used_tokens, rec_only: true)
    else
      dst.copy_from!(src)
    end
  }
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  max_seq = prompt_ids.size + gen_tokens + max_gamma + 8
  prefix_ids = prompt_ids[0, prompt_ids.size - 1]
  raise "GPU pipeline MTP K2 reject diagnostic requires at least two prompt tokens" if mtp_k2_on_reject_enabled && prefix_ids.empty?
  prompt_last_token = prompt_ids[-1]
  prompt_pos_last = prompt_ids.size - 1
  last_token = prompt_last_token
  pos_last = prompt_pos_last

  basis_size_bytes = (h_k * rank * s).to_i64 * sizeof(Float32)
  state_size_bytes = (h_v * s * rank).to_i64 * sizeof(Float32)
  full_state_size = h_v * s * s
  lowrank_set = Set(Int32).new(layer_bases.keys)
  if threshold = ProbeRuntime.gpu_draft_update_risk_threshold
    plan = planned_gpu_update_risk_offsets(weights, prompt_ids, gen_tokens,
      layer_bases, rank, prompt_ids.size - 1, threshold.not_nil!)
    exact_refresh_offsets = (exact_refresh_offsets + plan[:offsets]).uniq.sort
    total_steps = plan[:approx_steps] + plan[:fallback_steps]
    approx_rate = total_steps > 0 ? (100.0 * plan[:approx_steps] / total_steps) : 0.0
    puts "self_spec_gpu_update_risk_plan layers=#{lowrank_set.to_a.sort.join(',')} rank=#{rank} threshold=#{threshold} fallback_score=#{ProbeRuntime.fallback_score_mode} gen_tokens=#{gen_tokens} offsets=#{plan[:offsets].join(',')} offset_count=#{plan[:offsets].size} layer_offsets=#{format_layer_offsets(plan[:layer_offsets])} approx_steps=#{plan[:approx_steps]} fallback_steps=#{plan[:fallback_steps]} approx_rate=#{approx_rate.round(2)}% exact_refresh_offsets=#{exact_refresh_offsets.join(',')} exact_ids=#{plan[:exact_ids].join(',')}"
  end
  if threshold = ProbeRuntime.gpu_draft_update_risk_layer_threshold
    plan = planned_gpu_update_risk_offsets(weights, prompt_ids, gen_tokens,
      layer_bases, rank, prompt_ids.size - 1, threshold.not_nil!)
    plan[:layer_offsets].each do |il, offsets|
      offsets.each { |offset| (exact_refresh_layer_offsets[offset] ||= Set(Int32).new) << il }
    end
    total_steps = plan[:approx_steps] + plan[:fallback_steps]
    approx_rate = total_steps > 0 ? (100.0 * plan[:approx_steps] / total_steps) : 0.0
    layer_steps = plan[:layer_offsets].values.sum(&.size)
    layer_plan = exact_refresh_layer_offsets.keys.sort.map { |offset| "#{offset}:#{exact_refresh_layer_offsets[offset].to_a.sort.join(',')}" }.join(";")
    puts "self_spec_gpu_update_risk_layer_plan layers=#{lowrank_set.to_a.sort.join(',')} rank=#{rank} threshold=#{threshold} fallback_score=#{ProbeRuntime.fallback_score_mode} gen_tokens=#{gen_tokens} layer_offsets=#{format_layer_offsets(plan[:layer_offsets])} layer_refresh_steps=#{layer_steps} approx_steps=#{plan[:approx_steps]} fallback_steps=#{plan[:fallback_steps]} approx_rate=#{approx_rate.round(2)}% exact_refresh_layer_offsets=#{layer_plan} exact_ids=#{plan[:exact_ids].join(',')}"
  end
  shared_basis_bufs = {} of Int32 => ML::MetalBuffer
  layer_bases.each do |il, bs|
    buf = ML::MetalBuffer.new(basis_size_bytes)
    buf.write(flatten_basis_for_metal(bs, rank, h_k, s))
    shared_basis_bufs[il] = buf
  end
  if first_basis = shared_basis_bufs.values.first?
    warm_full_state = ML::MetalBuffer.new(full_state_size.to_i64 * sizeof(Float32))
    warm_full_state.contents.as(Pointer(UInt8)).clear(warm_full_state.size)
    ML::GGUF::Qwen35Metal.lowrank_project_state_buf(warm_full_state, first_basis,
      h_k, h_v, s, rank, command_queue_name: "self_spec_gpu_pipeline_draft")
  end
  updown_x_mean_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_c_mean_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_coeff_w_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_down_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_coeff_q8_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_coeff_q8_scale_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_down_q8_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_down_q8_scale_bufs = nil.as(Hash(Int32, ML::MetalBuffer)?)
  updown_actual_rank = 0
  if updown_rank = draft_updown_rank
    raise "GPU pipeline pca-updown cannot be combined with global draft_no_ffn" if draft_no_ffn
    raise "GPU pipeline pca-updown cannot be combined with draft_skip_recurrent_ffn" if draft_skip_recurrent_ffn
    if no_ffn_set = draft_no_ffn_layer_indices
      updown_set_for_conflict = draft_updown_layer_indices || lowrank_set
      overlap = no_ffn_set.select { |il| updown_set_for_conflict.includes?(il) }
      raise "GPU pipeline pca-updown/no-ffn layer sets overlap: #{overlap.to_a.sort.join(',')}" unless overlap.empty?
    end
    adapters = ffn_updown_adapters || raise "GPU pipeline pca-updown requires FFN up/down adapters"
    if draft_updown_q8_metal
      maps = build_updown_adapter_q8_buffer_maps(adapters, draft_updown_layer_indices || lowrank_set, updown_rank, hp.n_embd)
      updown_x_mean_bufs = maps[:x_mean]
      updown_c_mean_bufs = maps[:c_mean]
      updown_coeff_q8_bufs = maps[:coeff_q8]
      updown_coeff_q8_scale_bufs = maps[:coeff_scales]
      updown_down_q8_bufs = maps[:down_q8]
      updown_down_q8_scale_bufs = maps[:down_scales]
      updown_actual_rank = maps[:rank]
      puts "ffn_updown_pca_adapter_metal mode=raw_q8 layers=#{(draft_updown_layer_indices || lowrank_set).to_a.sort.join(',')} max_rank=#{updown_actual_rank}"
    else
      maps = build_updown_adapter_buffer_maps(adapters, draft_updown_layer_indices || lowrank_set, updown_rank, hp.n_embd)
      updown_x_mean_bufs = maps[:x_mean]
      updown_c_mean_bufs = maps[:c_mean]
      updown_coeff_w_bufs = maps[:coeff_w]
      updown_down_bufs = maps[:down]
      updown_actual_rank = maps[:rank]
    end
  end
  wba = WbaTrace.maybe("self_spec_gpu_pipeline")
  attr_collect = true
  draft_steps = 0
  draft_blocks = 0
  draft_fork_ms = 0.0
  draft_token_buf_ms = 0.0
  draft_lr_project_ms = 0.0
  draft_submit_ms = 0.0
  draft_commit_ms = 0.0
  draft_wait_block_ms = 0.0
  draft_read_ids_ms = 0.0
  draft_resync_ms = 0.0
  draft_resyncs = 0
  draft_wasted_tail_tokens = 0
  draft_wasted_next_tokens = 0
  verifier_initial_ms = 0.0
  verifier_prefill_ms = 0.0
  verifier_chunks = 0
  verifier_tokens_count = 0
  verifier_tail_skip_tokens = 0

  copy_owned_resync_base = ->(src : ML::GGUF::Qwen35CPU::State, used_tokens : Int32, label : String) {
    t_copy = Time.instant
    dst = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(dst, hp)
    copy_verifier_state.call(dst, src, used_tokens)
    draft_fork_ms += (Time.instant - t_copy).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "copy_resync_base_#{label}", t_copy, Time.instant))
    dst
  }
  copy_owned_resync_base_from_branch_snapshot = ->(current : ML::GGUF::Qwen35CPU::State,
                                                   snapshot : ML::GGUF::Qwen35CPU::State,
                                                   snapshot_pos : Int32,
                                                   base_used_tokens : Int32,
                                                   accepted_tail_prefix : Array(Int32),
                                                   label : String) {
    t_copy = Time.instant
    dst = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(dst, hp)
    # Avoid paying KV copies at the branch point. Current verifier KV rows
    # before the next seed token remain exact (or are replay-overwritten),
    # while recurrent state is rewound from the compact branch snapshot.
    copy_verifier_state.call(dst, current, base_used_tokens)
    copy_verifier_recurrent_state.call(dst, snapshot, snapshot_pos)
    unless accepted_tail_prefix.empty?
      ML::GGUF::Qwen35CPU.prefill_tokens(weights, accepted_tail_prefix, snapshot_pos, dst)
    end
    draft_fork_ms += (Time.instant - t_copy).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "copy_resync_base_#{label}", t_copy, Time.instant))
    dst
  }

  build_lr_states = ->(state : ML::GGUF::Qwen35CPU::State, block_cmd : ML::Metal::CommandBuffer?) {
    t_lr_build = Time.instant
    bufs = {} of Int32 => ML::MetalBuffer
    layer_bases.each do |il, bs|
      ssm_buf = state.layers[il].ssm_state_buf
      buf = if ssm_buf
              if cmd = block_cmd
                ML::GGUF::Qwen35Metal.lowrank_project_state_append(ssm_buf, shared_basis_bufs[il],
                  h_k, h_v, s, rank, cmd)
              else
                ML::GGUF::Qwen35Metal.lowrank_project_state_buf(ssm_buf, shared_basis_bufs[il],
                  h_k, h_v, s, rank, command_queue_name: "self_spec_gpu_pipeline_draft")
              end
            else
              cpu_buf = ML::MetalBuffer.new(state_size_bytes)
              full_state = state.layers[il].ssm_state ||= Array(Float32).new(full_state_size, 0.0_f32)
              cpu_buf.write(project_full_state_to_lowrank(full_state, bs, rank, h_k, h_v, s))
              cpu_buf
            end
      bufs[il] = buf
    end
    draft_lr_project_ms += (Time.instant - t_lr_build).total_milliseconds if attr_collect
    bufs
  }
  fresh_full_current = ->{
    current = {} of Int32 => Bool
    lowrank_set.each { |il| current[il] = true }
    current
  }

  submit_block = ->(state : ML::GGUF::Qwen35CPU::State, lr_bufs : Hash(Int32, ML::MetalBuffer), full_current : Hash(Int32, Bool), token_buf : ML::MetalBuffer, pos_start : Int32, label : String, block_cmd : ML::Metal::CommandBuffer?, steps : Int32, use_updown : Bool) {
    submissions = [] of ML::GGUF::Qwen35Metal::DecodeWaveSubmission
    if attr_collect
      draft_blocks += 1
      draft_steps += steps
    end
    cur_token_buf = token_buf
    split_tokens = draft_block_tokens || (ENV["QWEN35_DRAFT_BLOCK_TOKENS"]?.try(&.to_i?) || DEFAULT_SELF_SPEC_GPU_PIPELINE_DRAFT_BLOCK_TOKENS)
    current_cmd = block_cmd || ML::GGUF::Qwen35Metal.decode_wave_command_buffer("self_spec_gpu_pipeline_draft")
    steps.times do |j|
      if split_tokens > 0 && j > 0 && (j % split_tokens) == 0
        t_commit = Time.instant
        current_cmd.commit
        draft_commit_ms += (Time.instant - t_commit).total_milliseconds if attr_collect
        wba.try(&.mark("draft", "commit_#{label}_part_#{j // split_tokens}", t_commit, Time.instant))
        current_cmd = ML::GGUF::Qwen35Metal.decode_wave_command_buffer("self_spec_gpu_pipeline_draft")
      end
      draft_offset = pos_start + j - prompt_pos_last
      global_exact_refresh = (exact_refresh_interval > 0 && (draft_offset % exact_refresh_interval) == 0) || draft_offset < exact_refresh_prefix || exact_refresh_offsets.includes?(draft_offset)
      exact_refresh_layers = Set(Int32).new
      if global_exact_refresh
        lowrank_set.each { |il| exact_refresh_layers << il }
      elsif layer_set = exact_refresh_layer_offsets[draft_offset]?
        layer_set.each { |il| exact_refresh_layers << il if lowrank_set.includes?(il) }
      end
      active_lowrank_set = Set(Int32).new
      lowrank_set.each { |il| active_lowrank_set << il unless exact_refresh_layers.includes?(il) }
      unless exact_refresh_layers.empty?
        exact_refresh_layers.each do |il|
          unless full_current[il]? == true
            ssm_buf = state.layers[il].ssm_state_buf || raise "GPU draft exact refresh missing full SSM state buffer for layer #{il}"
            ML::GGUF::Qwen35Metal.lowrank_reconstruct_state_append(lr_bufs[il], shared_basis_bufs[il],
              ssm_buf, h_k, h_v, s, rank, current_cmd)
          end
        end
      end
      t_submit = Time.instant
      sub = if tree2_enabled
              ML::GGUF::Qwen35CPU.forward_self_draft_top2_from_token_buf_async(weights, cur_token_buf, 0, pos_start + j, state,
                active_lowrank_set, lr_bufs, shared_basis_bufs, rank,
                lowrank_skip_ffn: active_draft_no_ffn,
                skip_recurrent_ffn: draft_skip_recurrent_ffn,
                lowrank_skip_ffn_layer_indices: active_draft_no_ffn_layer_indices,
                lowrank_updown_x_mean_bufs: updown_x_mean_bufs,
                lowrank_updown_c_mean_bufs: updown_c_mean_bufs,
                lowrank_updown_coeff_w_bufs: updown_coeff_w_bufs,
                lowrank_updown_down_bufs: updown_down_bufs,
                lowrank_updown_coeff_q8_bufs: updown_coeff_q8_bufs,
                lowrank_updown_coeff_q8_scale_bufs: updown_coeff_q8_scale_bufs,
                lowrank_updown_down_q8_bufs: updown_down_q8_bufs,
                lowrank_updown_down_q8_scale_bufs: updown_down_q8_scale_bufs,
                lowrank_updown_rank: use_updown ? updown_actual_rank : 0,
                lowrank_updown_layer_indices: draft_updown_layer_indices,
                scratch_namespace: "#{label}_#{j}",
                append_command_buffer: current_cmd).not_nil!
            else
              ML::GGUF::Qwen35CPU.forward_self_draft_top1_from_token_buf_async(weights, cur_token_buf, 0, pos_start + j, state,
                active_lowrank_set, lr_bufs, shared_basis_bufs, rank,
                lowrank_skip_ffn: active_draft_no_ffn,
                skip_recurrent_ffn: draft_skip_recurrent_ffn,
                lowrank_skip_ffn_layer_indices: active_draft_no_ffn_layer_indices,
                lowrank_updown_x_mean_bufs: updown_x_mean_bufs,
                lowrank_updown_c_mean_bufs: updown_c_mean_bufs,
                lowrank_updown_coeff_w_bufs: updown_coeff_w_bufs,
                lowrank_updown_down_bufs: updown_down_bufs,
                lowrank_updown_coeff_q8_bufs: updown_coeff_q8_bufs,
                lowrank_updown_coeff_q8_scale_bufs: updown_coeff_q8_scale_bufs,
                lowrank_updown_down_q8_bufs: updown_down_q8_bufs,
                lowrank_updown_down_q8_scale_bufs: updown_down_q8_scale_bufs,
                lowrank_updown_rank: use_updown ? updown_actual_rank : 0,
                lowrank_updown_layer_indices: draft_updown_layer_indices,
                scratch_namespace: "#{label}_#{j}",
                append_command_buffer: current_cmd).not_nil!
            end
      draft_submit_ms += (Time.instant - t_submit).total_milliseconds if attr_collect
      wba.try(&.mark("draft", "submit_#{label}_#{j}", t_submit, Time.instant))
      submissions << sub
      exact_refresh_layers.each do |il|
        ssm_buf = state.layers[il].ssm_state_buf || raise "GPU draft exact refresh missing updated SSM state buffer for layer #{il}"
        lr_bufs[il] = ML::GGUF::Qwen35Metal.lowrank_project_state_append(ssm_buf, shared_basis_bufs[il],
          h_k, h_v, s, rank, current_cmd)
        full_current[il] = true
      end
      active_lowrank_set.each { |il| full_current[il] = false }
      cur_token_buf = sub.top1_id_buf.not_nil!
    end
    t_commit = Time.instant
    current_cmd.commit
    draft_commit_ms += (Time.instant - t_commit).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "commit_#{label}", t_commit, Time.instant))
    GpuDraftBlock.new(submissions, state, lr_bufs, full_current, use_updown,
      active_draft_no_ffn || !active_draft_no_ffn_layer_indices.nil?)
  }

  read_block = ->(block : GpuDraftBlock, limit : Int32, label : String) {
    active = block.submissions[0, limit]
    t_wait = Time.instant
    waited_cmds = Set(UInt64).new
    active.each do |sub|
      sub.pending_cmds.each do |cmd|
        id = cmd.object_id
        next if waited_cmds.includes?(id)
        cmd.wait
        waited_cmds << id
      end
      id = sub.cmd.object_id
      unless waited_cmds.includes?(id)
        sub.cmd.wait
        waited_cmds << id
      end
    end
    draft_wait_block_ms += (Time.instant - t_wait).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "wait_block_#{label}", t_wait, Time.instant))

    ids = Array(Int32).new(active.size)
    t_read = Time.instant
    active.each do |sub|
      ids << sub.top1_id_buf.not_nil!.contents.as(Pointer(UInt32)).value.to_i32
    end
    draft_read_ids_ms += (Time.instant - t_read).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "read_ids_#{label}", t_read, Time.instant))
    ids
  }

  read_second_id = ->(block : GpuDraftBlock, index : Int32) {
    if buf = block.submissions[index].second_id_buf
      buf.contents.as(Pointer(UInt32)).value.to_i32
    else
      -1_i32
    end
  }

  read_top2_margin = ->(block : GpuDraftBlock, index : Int32) {
    sub = block.submissions[index]
    if top = sub.top1_value_buf
      if second = sub.second_value_buf
        top.contents.as(Pointer(Float32)).value.to_f64 - second.contents.as(Pointer(Float32)).value.to_f64
      else
        nil
      end
    else
      nil
    end
  }
  branch_guard_snapshot_suffix_allowed = ->(block : GpuDraftBlock, suffix_start : Int32, verifier_size : Int32) {
    prefix_policy_enabled = !branch_guard_snapshot_prefix_suffix_thresholds.empty?
    selected_threshold = nil.as(Float64?)
    if prefix_policy_enabled
      branch_guard_snapshot_prefix_suffix_thresholds.each do |min_prefix, threshold|
        selected_threshold = threshold if suffix_start >= min_prefix
      end
    elsif threshold = branch_guard_snapshot_suffix_threshold
      selected_threshold = threshold
    end

    if threshold_value = selected_threshold
      min_threshold = branch_guard_snapshot_suffix_min_threshold
      suffix_end = Math.min(verifier_size, block.submissions.size) - 1
      if suffix_start > suffix_end
        false
      else
        allowed = false
        suffix_start.upto(suffix_end) do |i|
          if margin = read_top2_margin.call(block, i)
            min_ok = min_threshold.nil? || margin >= min_threshold.not_nil!
            if min_ok && margin <= threshold_value
              allowed = true
              break
            end
          end
        end
        allowed
      end
    else
      !prefix_policy_enabled
    end
  }
  branch_guard_split_allowed = ->(block : GpuDraftBlock, guard_index : Int32, verifier_size : Int32) {
    unless branch_guard_snapshot_only_split
      true
    else
      suffix_size = verifier_size - guard_index - 1
      snapshot_allowed = suffix_size > 0 &&
        guard_index + 1 >= branch_guard_snapshot_min_prefix &&
        branch_guard_snapshot_suffix_allowed.call(block, guard_index + 1, verifier_size)
      if snapshot_allowed
        true
      elsif threshold = branch_guard_no_snapshot_threshold
        if margin = read_top2_margin.call(block, guard_index)
          margin <= threshold.not_nil!
        else
          false
        end
      else
        false
      end
    end
  }

  drain_block = ->(block : GpuDraftBlock?) {
    if b = block
      waited_cmds = Set(UInt64).new
      b.submissions.each do |sub|
        sub.pending_cmds.each do |cmd|
          id = cmd.object_id
          next if waited_cmds.includes?(id)
          cmd.wait
          waited_cmds << id
        end
        id = sub.cmd.object_id
        unless waited_cmds.includes?(id)
          sub.cmd.wait
          waited_cmds << id
        end
      end
    end
  }

  draft_updown_agreement_checks = 0
  draft_updown_agreement_top1 = 0
  draft_updown_agreement_top2 = 0
  draft_updown_agreement_fails = 0
  draft_updown_agreement_probe_ms = 0.0
  agreement_margin_thresholds = draft_updown_agreement_margin_thresholds.uniq.sort
  agreement_margin_selected = Array(Int32).new(agreement_margin_thresholds.size, 0)
  agreement_margin_selected_passes = Array(Int32).new(agreement_margin_thresholds.size, 0)
  agreement_margin_selected_fails = Array(Int32).new(agreement_margin_thresholds.size, 0)
  agreement_margin_false_negatives = Array(Int32).new(agreement_margin_thresholds.size, 0)
  agreement_margin_count = 0
  agreement_margin_sum = 0.0
  agreement_margin_pass_count = 0
  agreement_margin_pass_sum = 0.0
  agreement_margin_fail_count = 0
  agreement_margin_fail_sum = 0.0

  probe_updown_agreement = ->(state : ML::GGUF::Qwen35CPU::State, token_buf : ML::MetalBuffer, pos_start : Int32, label : String, requested_steps : Int32) {
    t_probe = Time.instant
    probe_steps = Math.min(draft_updown_agreement_steps, requested_steps)

    low_state = state.fork
    low_lr_bufs = build_lr_states.call(low_state, nil)
    low_probe = submit_block.call(low_state, low_lr_bufs, fresh_full_current.call, token_buf, pos_start, "#{label}_agree_lowrank", nil, probe_steps, false)
    low_ids = read_block.call(low_probe, probe_steps, "#{label}_agree_lowrank")
    low_margin_min = Float64::INFINITY
    low_margin_checks = 0
    probe_steps.times do |i|
      if margin = read_top2_margin.call(low_probe, i)
        low_margin_min = margin if margin < low_margin_min
        low_margin_checks += 1
      end
    end

    up_state = state.fork
    up_lr_bufs = build_lr_states.call(up_state, nil)
    up_probe = submit_block.call(up_state, up_lr_bufs, fresh_full_current.call, token_buf, pos_start, "#{label}_agree_updown", nil, probe_steps, true)
    up_ids = read_block.call(up_probe, probe_steps, "#{label}_agree_updown")

    top1_match = true
    top2_match = false
    probe_steps.times do |i|
      if up_ids[i] == low_ids[i]
        next
      elsif i == 0 && (second_id = read_second_id.call(low_probe, i)) >= 0 && up_ids[i] == second_id
        top1_match = false
        top2_match = true
      else
        top1_match = false
        top2_match = false
        break
      end
    end
    if attr_collect
      passed = top1_match || top2_match
      draft_updown_agreement_checks += 1
      if top1_match
        draft_updown_agreement_top1 += 1
      elsif top2_match
        draft_updown_agreement_top2 += 1
      else
        draft_updown_agreement_fails += 1
      end
      if low_margin_checks > 0
        agreement_margin_count += 1
        agreement_margin_sum += low_margin_min
        if passed
          agreement_margin_pass_count += 1
          agreement_margin_pass_sum += low_margin_min
        else
          agreement_margin_fail_count += 1
          agreement_margin_fail_sum += low_margin_min
        end
        agreement_margin_thresholds.each_with_index do |threshold, ti|
          if low_margin_min >= threshold
            agreement_margin_selected[ti] += 1
            if passed
              agreement_margin_selected_passes[ti] += 1
            else
              agreement_margin_selected_fails[ti] += 1
            end
          elsif passed
            agreement_margin_false_negatives[ti] += 1
          end
        end
      end
      draft_updown_agreement_probe_ms += (Time.instant - t_probe).total_milliseconds
    end
    wba.try(&.mark("draft", "updown_agreement_#{label}", t_probe, Time.instant))
    top1_match || top2_match
  }

  submit_routed_block = ->(state : ML::GGUF::Qwen35CPU::State, lr_bufs : Hash(Int32, ML::MetalBuffer), full_current : Hash(Int32, Bool), token_buf : ML::MetalBuffer, pos_start : Int32, label : String, block_cmd : ML::Metal::CommandBuffer?, steps : Int32, requested_updown : Bool) {
    use_updown = requested_updown
    if requested_updown && draft_updown_agreement_gate && steps > 0
      use_updown = probe_updown_agreement.call(state, token_buf, pos_start, label, steps)
    end
    submit_block.call(state, lr_bufs, full_current, token_buf, pos_start, label, block_cmd, steps, use_updown)
  }

  submit_seed = ->(base_state : ML::GGUF::Qwen35CPU::State, token_id : Int32, pos_start : Int32, label : String, steps : Int32, use_updown : Bool) {
    t_fork = Time.instant
    state = base_state.fork
    draft_fork_ms += (Time.instant - t_fork).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "fork_#{label}", t_fork, Time.instant))
    t_token = Time.instant
    token_buf = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    token_buf.contents.as(Pointer(UInt32)).value = token_id.to_u32
    draft_token_buf_ms += (Time.instant - t_token).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "token_buf_#{label}", t_token, Time.instant))
    block_cmd = ML::GGUF::Qwen35Metal.decode_wave_command_buffer("self_spec_gpu_pipeline_draft")
    t_lr = Time.instant
    lr_bufs = build_lr_states.call(state, block_cmd)
    wba.try(&.mark("draft", "lr_states_#{label}", t_lr, Time.instant))
    submit_routed_block.call(state, lr_bufs, fresh_full_current.call, token_buf, pos_start, label, block_cmd, steps, use_updown)
  }

  submit_seed_owned = ->(state : ML::GGUF::Qwen35CPU::State, token_id : Int32, pos_start : Int32, label : String, steps : Int32, use_updown : Bool) {
    t_token = Time.instant
    token_buf = ML::MetalBuffer.new(sizeof(UInt32).to_i64)
    token_buf.contents.as(Pointer(UInt32)).value = token_id.to_u32
    draft_token_buf_ms += (Time.instant - t_token).total_milliseconds if attr_collect
    wba.try(&.mark("draft", "token_buf_#{label}_owned", t_token, Time.instant))
    block_cmd = ML::GGUF::Qwen35Metal.decode_wave_command_buffer("self_spec_gpu_pipeline_draft")
    t_lr = Time.instant
    lr_bufs = build_lr_states.call(state, block_cmd)
    wba.try(&.mark("draft", "lr_states_#{label}_owned", t_lr, Time.instant))
    submit_routed_block.call(state, lr_bufs, fresh_full_current.call, token_buf, pos_start, label, block_cmd, steps, use_updown)
  }

  state_before_last = verifier_state_after_prefix(weights, prefix_ids, max_seq)
  verifier_state = state_before_last.fork
  mtp_prev_hidden = nil.as(Array(Float32)?)
  mtp_input_hidden = nil.as(Array(Float32)?)
  mtp_input_token = last_token
  mtp_input_pos = pos_last
  if mtp_k2_on_reject_enabled
    prefix_hidden_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(prefix_hidden_state, hp)
    mtp_prev_hidden = ML::GGUF::Qwen35CPU.prefill_tokens_last_hidden(weights, prefix_ids, 0, prefix_hidden_state)
    # Keep the sparse reject diagnostic from charging one-time MTP Metal/BF16
    # setup to the first rare fallback call. Model/kernels are normally hot
    # before decode benchmarking; controller timing should reflect that.
    mtp_hidden_topk_for_fusion(weights, mtp_k2_on_reject.not_nil!, mtp_prev_hidden.not_nil!,
      last_token, pos_last, 2)
  end
  verifier_backup = if use_verifier_backup
                      backup = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
                      ML::GGUF::Qwen35CPU.prepare_state_metal!(backup, hp)
                      backup
                    else
                      nil
                    end
  branch_guard_snapshot_scratch = if branch_guard_snapshot_enabled
                                    snapshot = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
                                    ML::GGUF::Qwen35CPU.prepare_state_metal!(snapshot, hp)
                                    snapshot
                                  else
                                    nil
                                  end
  t_seed = Time.instant
  current_schedule_index = 0
  draft_updown_available = updown_actual_rank > 0
  draft_updown_chunks = 0
  draft_updown_cap_open = ->{
    if max_chunks = draft_updown_max_chunks
      draft_updown_chunks < max_chunks
    else
      true
    end
  }
  draft_updown_race_first_chunk = ProbeRuntime.self_spec_draft_updown_race_first_chunk
  draft_updown_first_margin_threshold = ProbeRuntime.self_spec_draft_updown_first_margin_threshold
  draft_updown_enabled = draft_updown_available && !draft_updown_race_first_chunk && draft_updown_first_margin_threshold.nil? && draft_updown_after_rejects <= 0 && draft_updown_after_full_accepts <= 0 && draft_updown_min_margin.nil? && draft_updown_cap_open.call
  draft_updown_full_accept_streak = 0
  draft_noffn_chunks = 0
  draft_noffn_full_accept_streak = 0
  draft_noffn_cap_open = ->{
    if max_chunks = draft_no_ffn_max_chunks
      draft_noffn_chunks < max_chunks
    else
      true
    end
  }
  enable_noffn_candidate = ->{
    active_draft_no_ffn = draft_no_ffn
    active_draft_no_ffn_layer_indices = draft_no_ffn_layer_indices
  }
  disable_noffn_candidate = ->{
    active_draft_no_ffn = false
    active_draft_no_ffn_layer_indices = nil
  }
  current_block = submit_seed.call(state_before_last, last_token, pos_last, "self_spec_seed", Math.min(schedule[current_schedule_index], gen_tokens), draft_updown_enabled)
  t_initial_target = Time.instant
  target_next_id = if mtp_k2_on_reject_enabled
                     hidden = ML::GGUF::Qwen35CPU.forward_hidden(weights, last_token, pos_last, verifier_state)
                     mtp_input_hidden = hidden
                     ML::GGUF::Qwen35CPU.hidden_top1(weights, hidden)[0]
                   else
                     ML::GGUF::Qwen35CPU.forward_top1(weights, last_token, pos_last, verifier_state)[0]
                   end
  verifier_initial_ms += (Time.instant - t_initial_target).total_milliseconds if attr_collect
  wba.try(&.mark("verifier", "initial_target", t_initial_target, Time.instant))
  current_proposal = read_block.call(current_block, Math.min(gamma, gen_tokens), "seed")
  if threshold = draft_updown_first_margin_threshold
    if draft_updown_available && !current_block.use_updown && gen_tokens > 0
      first_margin_min = Float64::INFINITY
      first_margin_checks = 0
      current_proposal.each_index do |i|
        if margin = read_top2_margin.call(current_block, i)
          first_margin_min = margin if margin < first_margin_min
          first_margin_checks += 1
        end
      end
      if first_margin_checks > 0 && first_margin_min <= threshold.not_nil!
        up_block = submit_seed.call(state_before_last, last_token, pos_last, "self_spec_seed_updown_first_margin", Math.min(schedule[current_schedule_index], gen_tokens), true)
        up_proposal = read_block.call(up_block, Math.min(gamma, gen_tokens), "seed_updown_first_margin")
        drain_block.call(current_block)
        current_block = up_block
        current_proposal = up_proposal
        draft_updown_enabled = true
      else
        draft_updown_enabled = false
      end
    end
  end
  if draft_updown_race_first_chunk && draft_updown_available && !current_block.use_updown && gen_tokens > 0
    score_first_chunk = ->(candidate_proposal : Array(Int32)) {
      eval_state = verifier_state.fork
      candidate_size = Math.min(candidate_proposal.size, gen_tokens)
      candidate = candidate_proposal[0, candidate_size]
      final_candidate = candidate_size >= gen_tokens
      verify_tokens = final_candidate && candidate.size > 1 ? candidate[0, candidate.size - 1] : candidate
      nexts = verify_tokens.empty? ? [] of {Int32, Float32} : ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, verify_tokens, prompt_ids.size, eval_state)
      expected = target_next_id
      accepted = 0
      reject_index = -1
      candidate.each_with_index do |cand, i|
        if cand == expected
          accepted += 1
          expected = nexts[i][0] if i < nexts.size
        else
          reject_index = i
          break
        end
      end
      {accepted, reject_index, candidate_size}
    }

    low_score = score_first_chunk.call(current_proposal)
    up_block = submit_seed.call(state_before_last, last_token, pos_last, "self_spec_seed_updown_race", Math.min(schedule[current_schedule_index], gen_tokens), true)
    up_proposal = read_block.call(up_block, Math.min(gamma, gen_tokens), "seed_updown_race")
    up_score = score_first_chunk.call(up_proposal)
    low_full = low_score[0] >= low_score[2] && low_score[1] < 0
    up_full = up_score[0] >= up_score[2] && up_score[1] < 0
    if up_full && !low_full
      drain_block.call(current_block)
      current_block = up_block
      current_proposal = up_proposal
      draft_updown_enabled = true
    else
      drain_block.call(up_block)
      draft_updown_enabled = false
    end
  end
  draft_seed_ms = (Time.instant - t_seed).total_milliseconds
  wba.try(&.mark("pipeline", "seed_block", t_seed, Time.instant))

  emitted_tokens = 0
  chunks = 0
  rejections = 0
  accepted_draft_tokens = 0
  proposed_tokens = 0
  draft_next_ms = 0.0
  verifier_ms = 0.0
  draft_wait_ms = 0.0
  backup_ms = 0.0
  rebuild_ms = 0.0
  controller_ms = 0.0
  replay_ms = 0.0
  overlap_ms = draft_seed_ms
  tree2_first_checks = 0
  tree2_first_rescues = 0
  tree2_first_misses = 0
  tree2_first_early_exits = 0
  tree2_anywhere_checks = 0
  tree2_anywhere_rescues = 0
  tree2_anywhere_misses = 0
  tree2_anywhere_early_exits = 0
  tree2_staged_checks = 0
  tree2_staged_rescues = 0
  tree2_staged_misses = 0
  tree2_staged_early_exits = 0
  tree2_staged_stages = 0
  tree2_margin_checks = 0
  tree2_margin_sum = 0.0
  tree2_margin_min = Float64::INFINITY
  tree2_reject_margin_checks = 0
  tree2_reject_margin_sum = 0.0
  tree2_reject_margin_min = Float64::INFINITY
  tree2_margin_guard_hits = 0
  tree2_margin_guard_tokens = 0
  tree2_margin_guard_rejects = 0
  tree2_margin_guard_passes = 0
  tree2_branch_guard_hits = 0
  tree2_branch_guard_tokens = 0
  tree2_branch_guard_rejects = 0
  tree2_branch_guard_rescues = 0
  tree2_branch_guard_misses = 0
  tree2_branch_guard_passes = 0
  tree2_branch_guard_prefix_rejects = 0
  tree2_branch_guard_replayless_resyncs = 0
  tree2_branch_guard_snapshot_copies = 0
  tree2_branch_guard_snapshot_ms = 0.0
  tree2_branch_guard_snapshot_restore_ms = 0.0
  tree2_branch_guard_snapshot_resync_base_ms = 0.0
  tree2_branch_guard_suffix_replays = 0
  tree2_branch_guard_suffix_replay_tokens = 0
  tree2_branch_guard_suffix_replay_ms = 0.0
  tree2_branch_guard_prefix_verify_ms = 0.0
  tree2_branch_guard_prefix_verify_tokens = 0
  tree2_branch_guard_token_verify_ms = 0.0
  tree2_branch_guard_token_verify_tokens = 0
  tree2_branch_guard_suffix_verify_ms = 0.0
  tree2_branch_guard_suffix_verify_tokens = 0
  tree2_branch_guard_snapshot_suffix_verify_ms = 0.0
  tree2_branch_guard_snapshot_suffix_verify_tokens = 0
  tree2_branch_guard_no_snapshot_suffix_verify_ms = 0.0
  tree2_branch_guard_no_snapshot_suffix_verify_tokens = 0
  risk_offramp_hits = 0
  risk_offramp_delayed_blocks = 0
  risk_offramp_delayed_tokens = 0
  mtp_k2_reject_checks = 0
  mtp_k2_reject_rescues = 0
  mtp_k2_reject_misses = 0
  mtp_k2_reject_ms = 0.0
  reject_offramp_hits = 0
  reject_offramp_tokens = 0
  reject_offramp_ms = 0.0
  record_tree2_margin = ->(margin : Float64) {
    tree2_margin_checks += 1
    tree2_margin_sum += margin
    tree2_margin_min = margin if margin < tree2_margin_min
  }
  record_tree2_reject_margin = ->(margin : Float64) {
    tree2_reject_margin_checks += 1
    tree2_reject_margin_sum += margin
    tree2_reject_margin_min = margin if margin < tree2_reject_margin_min
  }
  update_updown_after_accept = ->(margin_min : Float64, margin_checks : Int32) {
    if draft_updown_available
      if draft_updown_after_full_accepts > 0
        draft_updown_full_accept_streak += 1
      end
      if draft_updown_after_full_accepts > 0 || !draft_updown_min_margin.nil?
        margin_ok = if threshold = draft_updown_min_margin
                      margin_checks > 0 && margin_min >= threshold.not_nil!
                    else
                      true
                    end
        streak_ok = draft_updown_after_full_accepts <= 0 || draft_updown_full_accept_streak >= draft_updown_after_full_accepts
        draft_updown_enabled = margin_ok && streak_ok && draft_updown_cap_open.call
      end
    end
  }
  update_noffn_after_accept = ->(margin_min : Float64, margin_checks : Int32) {
    if draft_no_ffn_candidate
      draft_noffn_full_accept_streak += 1 if draft_no_ffn_after_full_accepts > 0
      if draft_no_ffn_dynamic_gate
        margin_ok = if threshold = draft_no_ffn_min_margin
                      margin_checks > 0 && margin_min >= threshold.not_nil!
                    else
                      true
                    end
        streak_ok = draft_no_ffn_after_full_accepts <= 0 || draft_noffn_full_accept_streak >= draft_no_ffn_after_full_accepts
        if margin_ok && streak_ok && draft_noffn_cap_open.call
          enable_noffn_candidate.call
        else
          disable_noffn_candidate.call
        end
      elsif !draft_no_ffn_max_chunks.nil? && !draft_noffn_cap_open.call
        disable_noffn_candidate.call
      end
    end
  }
  disable_updown_after_reject = ->(rejected_updown : Bool) {
    draft_updown_full_accept_streak = 0
    if rejected_updown
      draft_updown_enabled = false if draft_updown_fallback_on_reject || draft_updown_after_rejects > 0 || draft_updown_after_full_accepts > 0 || !draft_updown_min_margin.nil?
    elsif draft_updown_after_rejects > 0 && draft_updown_available && rejections >= draft_updown_after_rejects && draft_updown_cap_open.call
      draft_updown_enabled = true
    elsif (draft_updown_fallback_on_reject || draft_updown_after_full_accepts > 0 || !draft_updown_min_margin.nil?) && draft_updown_enabled
      draft_updown_enabled = false
    end
    draft_noffn_full_accept_streak = 0
    if draft_no_ffn_fallback_on_reject || draft_no_ffn_dynamic_gate
      disable_noffn_candidate.call
    end
  }
  exact_ids = [] of Int32
  emitted_ids = [] of Int32
  gamma_history = [] of Int32
  accept_history = [] of Int32
  reject_index_history = [] of Int32
  finish_reject_offramp = ->(label : String) {
    if reject_offramp_after <= 0 || rejections < reject_offramp_after || emitted_tokens >= gen_tokens
      false
    else
      t_offramp = Time.instant
      tokens_before = emitted_tokens
      reject_offramp_hits += 1
      while emitted_tokens < gen_tokens
        expected = target_next_id
        exact_ids << expected
        emitted_ids << expected
        last_token = expected
        pos_last = prompt_ids.size + emitted_tokens
        emitted_tokens += 1
        if emitted_tokens < gen_tokens
          target_next_id = ML::GGUF::Qwen35CPU.forward_top1(weights, last_token, pos_last, verifier_state)[0]
        end
      end
      emitted = emitted_tokens - tokens_before
      dt_offramp = (Time.instant - t_offramp).total_milliseconds
      reject_offramp_tokens += emitted
      reject_offramp_ms += dt_offramp
      verifier_ms += dt_offramp
      if attr_collect
        verifier_prefill_ms += dt_offramp
        verifier_chunks += Math.max(emitted - 1, 0)
        verifier_tokens_count += Math.max(emitted - 1, 0)
      end
      wba.try(&.mark("pipeline", "reject_offramp_#{label}", t_offramp, Time.instant))
      true
    end
  }

  while emitted_tokens < gen_tokens
    chunks += 1
    chunk_size = Math.min(current_proposal.size, gen_tokens - emitted_tokens)
    proposal = current_proposal[0, chunk_size]
    gamma_history << proposal.size
    proposed_tokens += proposal.size
    chunk_accepted_start = accepted_draft_tokens
    chunk_reject_index = -1
    draft_updown_chunks += 1 if current_block.use_updown
    draft_noffn_chunks += 1 if current_block.use_noffn
    draft_updown_enabled = false unless draft_updown_cap_open.call
    disable_noffn_candidate.call unless draft_noffn_cap_open.call
    cycle_start_pos = prompt_ids.size + emitted_tokens
    final_chunk = emitted_tokens + proposal.size >= gen_tokens
    verifier_tokens = final_chunk && proposal.size > 1 ? proposal[0, proposal.size - 1] : proposal
    proposal_margin_min = Float64::INFINITY
    proposal_margin_checks = 0
    if tree2_enabled
      proposal.each_index do |i|
        if margin = read_top2_margin.call(current_block, i)
          record_tree2_margin.call(margin)
          proposal_margin_min = margin if margin < proposal_margin_min
          proposal_margin_checks += 1
        end
      end
    end
    if attr_collect
      verifier_tail_skip_tokens += proposal.size - verifier_tokens.size
    end

    if tree2_staged_tokens > 0 && !proposal.empty?
      next_block = nil.as(GpuDraftBlock?)
      next_proposal_limit = 0
      chunk_draft_next_ms = 0.0
      next_schedule_index = current_schedule_index
      t_staged = Time.instant

      if emitted_tokens + proposal.size < gen_tokens
        t_next = Time.instant
        last_proposed_buf = current_block.submissions[proposal.size - 1].top1_id_buf.not_nil!
        next_schedule_index = (current_schedule_index + 1) % schedule.size
        next_steps = Math.min(schedule[next_schedule_index], gen_tokens - emitted_tokens - proposal.size)
        next_proposal_limit = next_steps
        next_block = submit_routed_block.call(current_block.state, current_block.lr_bufs, current_block.full_current, last_proposed_buf, pos_last + proposal.size, "self_spec_staged_next_#{chunks}", nil, next_steps, draft_updown_enabled)
        chunk_draft_next_ms += (Time.instant - t_next).total_milliseconds
      end

      chunk_emitted_start = emitted_tokens
      stage_offset = 0
      expected = target_next_id
      rejected = false
      while stage_offset < proposal.size && emitted_tokens < gen_tokens
        stage_size = Math.min(tree2_staged_tokens, proposal.size - stage_offset)
        stage_pos = cycle_start_pos + stage_offset
        stage_final_token = chunk_emitted_start + stage_offset + stage_size >= gen_tokens
        stage_verify_size = stage_final_token ? Math.max(stage_size - 1, 0) : stage_size

        if use_verifier_backup
          t_backup = Time.instant
          copy_verifier_state.call(verifier_backup.not_nil!, verifier_state, stage_pos)
          backup_ms += (Time.instant - t_backup).total_milliseconds
          wba.try(&.mark("controller", "staged_backup_#{chunks}_#{stage_offset}", t_backup, Time.instant))
        end

        t_verify = Time.instant
        stage_verify_tokens = stage_verify_size > 0 ? proposal[stage_offset, stage_verify_size] : [] of Int32
        target_nexts = stage_verify_tokens.empty? ? [] of {Int32, Float32} : ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, stage_verify_tokens, stage_pos, verifier_state)
        dt_verify = (Time.instant - t_verify).total_milliseconds
        verifier_ms += dt_verify
        if attr_collect
          verifier_prefill_ms += dt_verify
          verifier_chunks += 1
          verifier_tokens_count += stage_verify_tokens.size
          tree2_staged_stages += 1
        end
        wba.try(&.mark("verifier", "staged_chunk_#{chunks}_#{stage_offset}", t_verify, Time.instant))

        correction_or_accepted = [] of Int32
        stage_rejected = false
        stage_size.times do |j|
          cand = proposal[stage_offset + j]
          tree2_staged_checks += 1
          exact_ids << expected
          emitted = if cand == expected
                      accepted_draft_tokens += 1
                      cand
                    else
                      if margin = read_top2_margin.call(current_block, stage_offset + j)
                        record_tree2_reject_margin.call(margin)
                      end
                      second_id = read_second_id.call(current_block, stage_offset + j)
                      tree2_staged_rescues += 1 if second_id == expected
                      tree2_staged_misses += 1 if second_id != expected
                      tree2_staged_early_exits += 1
                      draft_wasted_tail_tokens += proposal.size - (stage_offset + j) - 1
                      rejections += 1
                      chunk_reject_index = stage_offset + j
                      rejected = true
                      stage_rejected = true
                      expected
                    end
          correction_or_accepted << emitted
          emitted_ids << emitted
          emitted_tokens += 1
          pos = stage_pos + j
          last_token = emitted
          pos_last = pos
          expected = target_nexts[j][0] if cand == emitted && j < target_nexts.size
          break if stage_rejected || emitted_tokens >= gen_tokens
        end

        if stage_rejected
          draft_wasted_next_tokens += next_proposal_limit if next_block
          drain_block.call(next_block)
          disable_updown_after_reject.call(current_block.use_updown)
          resync_base = nil.as(ML::GGUF::Qwen35CPU::State?)
          if emitted_tokens < gen_tokens
            if use_verifier_backup
              backup = verifier_backup.not_nil!
              copy_verifier_state.call(verifier_state, backup, stage_pos)
              t_replay = Time.instant
              corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, correction_or_accepted, stage_pos, verifier_state)
              replay_ms += (Time.instant - t_replay).total_milliseconds
              target_next_id = corrected[-1][0]
              resync_base = copy_owned_resync_base.call(backup, stage_pos, "staged_#{chunks}_#{stage_offset}")
              if correction_or_accepted.size > 1
                ML::GGUF::Qwen35CPU.prefill_tokens(weights, correction_or_accepted[0, correction_or_accepted.size - 1], stage_pos, resync_base)
              end
            else
              t_rebuild = Time.instant
              consumed = prompt_ids + emitted_ids
              verifier_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
              ML::GGUF::Qwen35CPU.prepare_state_metal!(verifier_state, hp)
              target_next_id = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, consumed, 0, verifier_state)[0]
              base_tokens = consumed[0, consumed.size - 1]
              resync_base = verifier_state_after_prefix(weights, base_tokens, max_seq)
              rebuild_ms += (Time.instant - t_rebuild).total_milliseconds
              wba.try(&.mark("controller", "staged_rebuild_#{chunks}", t_rebuild, Time.instant))
            end

            current_schedule_index = 0
            t_resync = Time.instant
            draft_resyncs += 1
            current_block = submit_seed_owned.call(resync_base.not_nil!, last_token, pos_last, "self_spec_staged_resync_#{chunks}", Math.min(schedule[current_schedule_index], gen_tokens - emitted_tokens), draft_updown_enabled)
            current_proposal = read_block.call(current_block, Math.min(schedule[current_schedule_index], gen_tokens - emitted_tokens), "staged_resync_#{chunks}")
            dt_resync = (Time.instant - t_resync).total_milliseconds
            draft_seed_ms += dt_resync
            draft_resync_ms += dt_resync if attr_collect
            wba.try(&.mark("pipeline", "staged_resync_#{chunks}", t_resync, Time.instant))
          end
          break
        end

        stage_offset += stage_size
      end

      unless rejected
        if next_block
          t_wait = Time.instant
          next_limit = Math.min(next_proposal_limit, gen_tokens - emitted_tokens)
          next_proposal = read_block.call(next_block.not_nil!, next_limit, "staged_next_#{chunks}")
          draft_wait_ms += (Time.instant - t_wait).total_milliseconds
          chunk_draft_next_ms += (Time.instant - t_wait).total_milliseconds
        else
          next_proposal = [] of Int32
        end
        update_updown_after_accept.call(proposal_margin_min, proposal_margin_checks)
        update_noffn_after_accept.call(proposal_margin_min, proposal_margin_checks)
        if emitted_tokens < gen_tokens
          target_next_id = expected
          draft_next_ms += chunk_draft_next_ms
          current_block = next_block.not_nil!
          current_proposal = next_proposal
          current_schedule_index = next_schedule_index
        end
      end

      overlap_ms += (Time.instant - t_staged).total_milliseconds
      wba.try(&.mark("pipeline", "tree2_staged_#{chunks}", t_staged, Time.instant))
      accept_history << (accepted_draft_tokens - chunk_accepted_start)
      reject_index_history << chunk_reject_index
      next
    end

    if tree2_anywhere && !proposal.empty?
      t_tree2_anywhere = Time.instant
      rejected = false
      proposal.each_with_index do |cand, i|
        expected = target_next_id
        tree2_anywhere_checks += 1
        exact_ids << expected
        emitted = if cand == expected
                    accepted_draft_tokens += 1
                    cand
                  else
                    if margin = read_top2_margin.call(current_block, i)
                      record_tree2_reject_margin.call(margin)
                    end
                    second_id = read_second_id.call(current_block, i)
                    tree2_anywhere_rescues += 1 if second_id == expected
                    tree2_anywhere_misses += 1 if second_id != expected
                    tree2_anywhere_early_exits += 1
                    draft_wasted_tail_tokens += proposal.size - i - 1
                    rejections += 1
                    chunk_reject_index = i
                    rejected = true
                    expected
                  end
        emitted_ids << emitted
        emitted_tokens += 1
        pos = cycle_start_pos + i
        last_token = emitted
        pos_last = pos

        if emitted_tokens < gen_tokens
          resync_base = rejected ? copy_owned_resync_base.call(verifier_state, pos, "tree2_anywhere_#{chunks}_#{i}") : nil
          t_verify = Time.instant
          target_next_id = ML::GGUF::Qwen35CPU.forward_top1(weights, emitted, pos, verifier_state)[0]
          dt_verify = (Time.instant - t_verify).total_milliseconds
          verifier_ms += dt_verify
          if attr_collect
            verifier_prefill_ms += dt_verify
            verifier_chunks += 1
            verifier_tokens_count += 1
          end

          if rejected
            disable_updown_after_reject.call(current_block.use_updown)
            current_schedule_index = 0
            t_resync = Time.instant
            draft_resyncs += 1
            current_block = submit_seed_owned.call(resync_base.not_nil!, last_token, pos_last, "self_spec_tree2_anywhere_#{chunks}", Math.min(schedule[current_schedule_index], gen_tokens - emitted_tokens), draft_updown_enabled)
            current_proposal = read_block.call(current_block, Math.min(schedule[current_schedule_index], gen_tokens - emitted_tokens), "tree2_anywhere_#{chunks}")
            dt_resync = (Time.instant - t_resync).total_milliseconds
            draft_seed_ms += dt_resync
            draft_resync_ms += dt_resync if attr_collect
          end
        end
        break if rejected || emitted_tokens >= gen_tokens
      end

      unless rejected
        update_updown_after_accept.call(proposal_margin_min, proposal_margin_checks)
        update_noffn_after_accept.call(proposal_margin_min, proposal_margin_checks)
        if emitted_tokens < gen_tokens
          t_next = Time.instant
          current_schedule_index = (current_schedule_index + 1) % schedule.size
          next_steps = Math.min(schedule[current_schedule_index], gen_tokens - emitted_tokens)
          last_proposed_buf = current_block.submissions[proposal.size - 1].top1_id_buf.not_nil!
          current_block = submit_routed_block.call(current_block.state, current_block.lr_bufs, current_block.full_current, last_proposed_buf, pos_last, "self_spec_tree2_anywhere_next_#{chunks}", nil, next_steps, draft_updown_enabled)
          current_proposal = read_block.call(current_block, next_steps, "tree2_anywhere_next_#{chunks}")
          draft_next_ms += (Time.instant - t_next).total_milliseconds
        end
      end
      overlap_ms += (Time.instant - t_tree2_anywhere).total_milliseconds
      wba.try(&.mark("pipeline", "tree2_anywhere_#{chunks}", t_tree2_anywhere, Time.instant))
      accept_history << (accepted_draft_tokens - chunk_accepted_start)
      reject_index_history << chunk_reject_index
      next
    end

    if tree2_first && !proposal.empty? && proposal[0] != target_next_id
      t_tree2 = Time.instant
      tree2_first_checks += 1
      if margin = read_top2_margin.call(current_block, 0)
        record_tree2_reject_margin.call(margin)
      end
      second_id = read_second_id.call(current_block, 0)
      expected = target_next_id
      tree2_first_rescues += 1 if second_id == expected
      tree2_first_misses += 1 if second_id != expected
      tree2_first_early_exits += 1
      draft_wasted_tail_tokens += proposal.size - 1
      rejections += 1
      chunk_reject_index = 0
      disable_updown_after_reject.call(current_block.use_updown)
      exact_ids << expected
      emitted_ids << expected
      emitted_tokens += 1
      last_token = expected
      pos_last = cycle_start_pos

      if emitted_tokens < gen_tokens
        resync_base = copy_owned_resync_base.call(verifier_state, cycle_start_pos, "tree2_first_#{chunks}")
        t_verify = Time.instant
        target_next_id = ML::GGUF::Qwen35CPU.forward_top1(weights, expected, cycle_start_pos, verifier_state)[0]
        dt_verify = (Time.instant - t_verify).total_milliseconds
        verifier_ms += dt_verify
        if attr_collect
          verifier_prefill_ms += dt_verify
          verifier_chunks += 1
          verifier_tokens_count += 1
        end

        if finish_reject_offramp.call("tree2_first_#{chunks}")
          overlap_ms += (Time.instant - t_tree2).total_milliseconds
          wba.try(&.mark("pipeline", "tree2_first_#{chunks}", t_tree2, Time.instant))
          accept_history << (accepted_draft_tokens - chunk_accepted_start)
          reject_index_history << chunk_reject_index
          next
        end

        current_schedule_index = 0
        t_resync = Time.instant
        draft_resyncs += 1
        current_block = submit_seed_owned.call(resync_base, last_token, pos_last, "self_spec_tree2_first_#{chunks}", Math.min(schedule[current_schedule_index], gen_tokens - emitted_tokens), draft_updown_enabled)
        current_proposal = read_block.call(current_block, Math.min(schedule[current_schedule_index], gen_tokens - emitted_tokens), "tree2_first_#{chunks}")
        dt_resync = (Time.instant - t_resync).total_milliseconds
        draft_seed_ms += dt_resync
        draft_resync_ms += dt_resync if attr_collect
      end
      overlap_ms += (Time.instant - t_tree2).total_milliseconds
      wba.try(&.mark("pipeline", "tree2_first_#{chunks}", t_tree2, Time.instant))
      accept_history << (accepted_draft_tokens - chunk_accepted_start)
      reject_index_history << chunk_reject_index
      next
    elsif tree2_first
      tree2_first_checks += 1
    end

    next_block = nil.as(GpuDraftBlock?)
    next_proposal_limit = 0
    chunk_draft_next_ms = 0.0
    t_overlap = Time.instant
    risk_offramp = false
    unless risk_offramp_margin.nil?
      risk_threshold : Float64 = risk_offramp_margin.not_nil!
      verifier_tokens.each_index do |i|
        if margin = read_top2_margin.call(current_block, i)
          if margin <= risk_threshold
            risk_offramp = true
            risk_offramp_hits += 1
            break
          end
        end
      end
    end
    branch_guard_index = nil.as(Int32?)
    if branch_threshold = tree2_branch_guard
      if !branch_guard_until_reject || rejections == 0
        verifier_tokens.each_index do |i|
          if margin = read_top2_margin.call(current_block, i)
            if margin <= branch_threshold
              if branch_guard_split_allowed.call(current_block, i, verifier_tokens.size)
                branch_guard_index = i
                break
              end
            end
          end
        end
      end
    end
    refresh_current_draft = draft_refresh_on_accept || (draft_updown_refresh_on_accept && current_block.use_updown)
    can_overlap_next_after_branch_guard = branch_guard_index.nil? || branch_guard_overlap_next
    if emitted_tokens + proposal.size < gen_tokens && !risk_offramp && !refresh_current_draft && can_overlap_next_after_branch_guard
      t_next = Time.instant
      last_proposed_buf = current_block.submissions[proposal.size - 1].top1_id_buf.not_nil!
      next_schedule_index = (current_schedule_index + 1) % schedule.size
      next_steps = Math.min(schedule[next_schedule_index], gen_tokens - emitted_tokens - proposal.size)
      next_proposal_limit = next_steps
      next_block = submit_routed_block.call(current_block.state, current_block.lr_bufs, current_block.full_current, last_proposed_buf, pos_last + proposal.size, "self_spec_next_#{chunks}", nil, next_steps, draft_updown_enabled)
      chunk_draft_next_ms += (Time.instant - t_next).total_milliseconds
    else
      next_schedule_index = (emitted_tokens + proposal.size < gen_tokens) ? ((current_schedule_index + 1) % schedule.size) : current_schedule_index
      if risk_offramp && emitted_tokens + proposal.size < gen_tokens
        next_proposal_limit = Math.min(schedule[next_schedule_index], gen_tokens - emitted_tokens - proposal.size)
        risk_offramp_delayed_blocks += 1
        risk_offramp_delayed_tokens += next_proposal_limit
      end
    end

    if use_verifier_backup
      t_backup = Time.instant
      copy_verifier_state.call(verifier_backup.not_nil!, verifier_state, cycle_start_pos)
      backup_ms += (Time.instant - t_backup).total_milliseconds
      wba.try(&.mark("controller", "backup_#{chunks}", t_backup, Time.instant))
    end
    guard_index = nil.as(Int32?)
    if guard_threshold = tree2_margin_guard
      verifier_tokens.each_index do |i|
        if margin = read_top2_margin.call(current_block, i)
          if margin <= guard_threshold
            guard_index = i
            break
          end
        end
      end
    end

    guard_rejected = false
    branch_guard_resync_ready = false
    branch_guard_resync_index = -1
    target_nexts = [] of {Int32, Float32}
    target_hiddens = [] of Float32
    branch_guard_snapshot_state = nil.as(ML::GGUF::Qwen35CPU::State?)
    branch_guard_snapshot_pos = -1
    if bgi = branch_guard_index
      tree2_branch_guard_hits += 1
      tree2_branch_guard_tokens += bgi + 1
      handled_branch_guard = false
      single_pass_checkpoint = branch_guard_single_pass_checkpoint &&
                               branch_guard_snapshot_enabled &&
                               verifier_tokens.size > bgi + 1 &&
                               bgi + 1 >= branch_guard_snapshot_min_prefix &&
                               branch_guard_snapshot_suffix_allowed.call(current_block, bgi + 1, verifier_tokens.size)
      if single_pass_checkpoint
        snapshot = branch_guard_snapshot_scratch.not_nil!
        branch_guard_snapshot_pos = cycle_start_pos + bgi + 1
        t_verify_onepass = Time.instant
        target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s_recurrent_checkpoint(weights, verifier_tokens, cycle_start_pos,
          verifier_state, bgi, snapshot)
        dt_verify_onepass = (Time.instant - t_verify_onepass).total_milliseconds
        verifier_ms += dt_verify_onepass
        tree2_branch_guard_suffix_verify_ms += dt_verify_onepass
        tree2_branch_guard_suffix_verify_tokens += verifier_tokens.size
        tree2_branch_guard_snapshot_suffix_verify_ms += dt_verify_onepass
        tree2_branch_guard_snapshot_suffix_verify_tokens += verifier_tokens.size
        if attr_collect
          verifier_prefill_ms += dt_verify_onepass
          verifier_chunks += 1
          verifier_tokens_count += verifier_tokens.size
        end
        wba.try(&.mark("verifier", "branch_guard_onepass_checkpoint_#{chunks}", t_verify_onepass, Time.instant))

        expected_branch = target_next_id
        prefix_ok = true
        bgi.times do |i|
          cand = proposal[i]
          if cand == expected_branch
            expected_branch = target_nexts[i][0]
          else
            tree2_branch_guard_prefix_rejects += 1
            prefix_ok = false
            break
          end
        end

        if prefix_ok
          guard_expected = expected_branch
          guard_cand = proposal[bgi]
          if guard_cand == guard_expected
            tree2_branch_guard_passes += 1
            branch_guard_snapshot_state = snapshot
            tree2_branch_guard_snapshot_copies += 1
          else
            second_id = read_second_id.call(current_block, bgi)
            tree2_branch_guard_rejects += 1
            if second_id == guard_expected
              tree2_branch_guard_rescues += 1
            else
              tree2_branch_guard_misses += 1
            end
          end
        end
        handled_branch_guard = true
      end

      unless handled_branch_guard
      prefix_size = bgi
      prefix_ok = true
      expected_branch = target_next_id
      if prefix_size > 0
        t_verify_prefix = Time.instant
        prefix_tokens = proposal[0, prefix_size]
        target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, prefix_tokens, cycle_start_pos, verifier_state)
        dt_verify_prefix = (Time.instant - t_verify_prefix).total_milliseconds
        verifier_ms += dt_verify_prefix
        tree2_branch_guard_prefix_verify_ms += dt_verify_prefix
        tree2_branch_guard_prefix_verify_tokens += prefix_tokens.size
        if attr_collect
          verifier_prefill_ms += dt_verify_prefix
          verifier_chunks += 1
          verifier_tokens_count += prefix_tokens.size
        end
        wba.try(&.mark("verifier", "branch_guard_prefix_#{chunks}", t_verify_prefix, Time.instant))

        prefix_size.times do |i|
          cand = proposal[i]
          if cand == expected_branch
            expected_branch = target_nexts[i][0]
          else
            if margin = read_top2_margin.call(current_block, i)
              record_tree2_reject_margin.call(margin)
            end
            tree2_branch_guard_prefix_rejects += 1
            prefix_ok = false
            break
          end
        end
      end

      if prefix_ok
        guard_expected = expected_branch
        guard_cand = proposal[bgi]
        if guard_cand == guard_expected
          tree2_branch_guard_passes += 1
          if branch_guard_snapshot_enabled
            t_verify_guard = Time.instant
            guard_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, [guard_cand], cycle_start_pos + bgi, verifier_state)
            target_nexts.concat(guard_nexts)
            dt_verify_guard = (Time.instant - t_verify_guard).total_milliseconds
            verifier_ms += dt_verify_guard
            tree2_branch_guard_token_verify_ms += dt_verify_guard
            tree2_branch_guard_token_verify_tokens += 1
            if attr_collect
              verifier_prefill_ms += dt_verify_guard
              verifier_chunks += 1
              verifier_tokens_count += 1
            end
            wba.try(&.mark("verifier", "branch_guard_token_#{chunks}", t_verify_guard, Time.instant))

            suffix_size = verifier_tokens.size - bgi - 1
            if suffix_size > 0 && bgi + 1 >= branch_guard_snapshot_min_prefix && branch_guard_snapshot_suffix_allowed.call(current_block, bgi + 1, verifier_tokens.size)
              t_snapshot = Time.instant
              snapshot = branch_guard_snapshot_scratch.not_nil!
              branch_guard_snapshot_pos = cycle_start_pos + bgi + 1
              copy_verifier_recurrent_state.call(snapshot, verifier_state, branch_guard_snapshot_pos)
              branch_guard_snapshot_state = snapshot
              dt_snapshot = (Time.instant - t_snapshot).total_milliseconds
              tree2_branch_guard_snapshot_copies += 1
              tree2_branch_guard_snapshot_ms += dt_snapshot
              wba.try(&.mark("verifier", "branch_guard_snapshot_#{chunks}", t_snapshot, Time.instant))

              t_verify_suffix = Time.instant
              suffix_tokens = verifier_tokens[bgi + 1, suffix_size]
              suffix_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, suffix_tokens, cycle_start_pos + bgi + 1, verifier_state)
              target_nexts.concat(suffix_nexts)
              dt_verify_suffix = (Time.instant - t_verify_suffix).total_milliseconds
              verifier_ms += dt_verify_suffix
              tree2_branch_guard_suffix_verify_ms += dt_verify_suffix
              tree2_branch_guard_suffix_verify_tokens += suffix_tokens.size
              tree2_branch_guard_snapshot_suffix_verify_ms += dt_verify_suffix
              tree2_branch_guard_snapshot_suffix_verify_tokens += suffix_tokens.size
              if attr_collect
                verifier_prefill_ms += dt_verify_suffix
                verifier_chunks += 1
                verifier_tokens_count += suffix_tokens.size
              end
              wba.try(&.mark("verifier", "branch_guard_suffix_#{chunks}", t_verify_suffix, Time.instant))
            elsif suffix_size > 0
              t_verify_suffix = Time.instant
              suffix_tokens = verifier_tokens[bgi + 1, suffix_size]
              suffix_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, suffix_tokens, cycle_start_pos + bgi + 1, verifier_state)
              target_nexts.concat(suffix_nexts)
              dt_verify_suffix = (Time.instant - t_verify_suffix).total_milliseconds
              verifier_ms += dt_verify_suffix
              tree2_branch_guard_suffix_verify_ms += dt_verify_suffix
              tree2_branch_guard_suffix_verify_tokens += suffix_tokens.size
              tree2_branch_guard_no_snapshot_suffix_verify_ms += dt_verify_suffix
              tree2_branch_guard_no_snapshot_suffix_verify_tokens += suffix_tokens.size
              if attr_collect
                verifier_prefill_ms += dt_verify_suffix
                verifier_chunks += 1
                verifier_tokens_count += suffix_tokens.size
              end
              wba.try(&.mark("verifier", "branch_guard_suffix_no_snapshot_#{chunks}", t_verify_suffix, Time.instant))
            end
          else
            suffix_size = verifier_tokens.size - bgi
            if suffix_size > 0
              t_verify_suffix = Time.instant
              suffix_tokens = verifier_tokens[bgi, suffix_size]
              suffix_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, suffix_tokens, cycle_start_pos + bgi, verifier_state)
              target_nexts.concat(suffix_nexts)
              dt_verify_suffix = (Time.instant - t_verify_suffix).total_milliseconds
              verifier_ms += dt_verify_suffix
              tree2_branch_guard_suffix_verify_ms += dt_verify_suffix
              tree2_branch_guard_suffix_verify_tokens += suffix_tokens.size
              tree2_branch_guard_no_snapshot_suffix_verify_ms += dt_verify_suffix
              tree2_branch_guard_no_snapshot_suffix_verify_tokens += suffix_tokens.size
              if attr_collect
                verifier_prefill_ms += dt_verify_suffix
                verifier_chunks += 1
                verifier_tokens_count += suffix_tokens.size
              end
              wba.try(&.mark("verifier", "branch_guard_suffix_#{chunks}", t_verify_suffix, Time.instant))
            end
          end
        else
          if margin = read_top2_margin.call(current_block, bgi)
            record_tree2_reject_margin.call(margin)
          end
          second_id = read_second_id.call(current_block, bgi)
          tree2_branch_guard_rejects += 1
          if second_id == guard_expected
            tree2_branch_guard_rescues += 1
          else
            tree2_branch_guard_misses += 1
          end
          branch_guard_resync_ready = true
          branch_guard_resync_index = bgi
        end
      end
      end
    elsif gi = guard_index
      guard_verify_size = gi + 1
      tree2_margin_guard_hits += 1
      tree2_margin_guard_tokens += guard_verify_size
      t_verify = Time.instant
      guard_tokens = proposal[0, guard_verify_size]
      target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, guard_tokens, cycle_start_pos, verifier_state)
      dt_verify = (Time.instant - t_verify).total_milliseconds
      verifier_ms += dt_verify
      if attr_collect
        verifier_prefill_ms += dt_verify
        verifier_chunks += 1
        verifier_tokens_count += guard_tokens.size
      end
      wba.try(&.mark("verifier", "margin_guard_prefix_#{chunks}", t_verify, Time.instant))

      expected_guard = target_next_id
      guard_verify_size.times do |i|
        cand = proposal[i]
        if cand == expected_guard
          expected_guard = target_nexts[i][0]
        else
          if margin = read_top2_margin.call(current_block, i)
            record_tree2_reject_margin.call(margin)
          end
          tree2_margin_guard_rejects += 1
          guard_rejected = true
          break
        end
      end

      unless guard_rejected
        tree2_margin_guard_passes += 1
        suffix_size = verifier_tokens.size - guard_verify_size
        if suffix_size > 0
          t_verify_suffix = Time.instant
          suffix_tokens = verifier_tokens[guard_verify_size, suffix_size]
          suffix_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, suffix_tokens, cycle_start_pos + guard_verify_size, verifier_state)
          target_nexts.concat(suffix_nexts)
          dt_verify_suffix = (Time.instant - t_verify_suffix).total_milliseconds
          verifier_ms += dt_verify_suffix
          if attr_collect
            verifier_prefill_ms += dt_verify_suffix
            verifier_chunks += 1
            verifier_tokens_count += suffix_tokens.size
          end
          wba.try(&.mark("verifier", "margin_guard_suffix_#{chunks}", t_verify_suffix, Time.instant))
        end
      end
    else
      t_verify = Time.instant
      if mtp_k2_on_reject_enabled && !verifier_tokens.empty?
        hidden_top1s = ML::GGUF::Qwen35CPU.prefill_tokens_hidden_top1s(weights, verifier_tokens, cycle_start_pos, verifier_state)
        target_hiddens = hidden_top1s[:hidden]
        target_nexts = hidden_top1s[:top1s]
      else
        target_nexts = verifier_tokens.empty? ? [] of {Int32, Float32} : ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, verifier_tokens, cycle_start_pos, verifier_state)
      end
      dt_verify = (Time.instant - t_verify).total_milliseconds
      verifier_ms += dt_verify
      if attr_collect
        verifier_prefill_ms += dt_verify
        verifier_chunks += 1
        verifier_tokens_count += verifier_tokens.size
      end
      wba.try(&.mark("verifier", "chunk_#{chunks}", t_verify, Time.instant))
    end

    if next_block && !guard_rejected
      t_wait = Time.instant
      next_limit = Math.min(next_proposal_limit, gen_tokens - emitted_tokens - proposal.size)
      next_proposal = read_block.call(next_block.not_nil!, next_limit, "next_#{chunks}")
      draft_wait_ms += (Time.instant - t_wait).total_milliseconds
      chunk_draft_next_ms += (Time.instant - t_wait).total_milliseconds
    else
      next_proposal = [] of Int32
    end
    overlap_ms += (Time.instant - t_overlap).total_milliseconds
    wba.try(&.mark("pipeline", "overlap_chunk_#{chunks}", t_overlap, Time.instant))

    t_controller = Time.instant
    correction_or_accepted = [] of Int32
    expected = target_next_id
    rejected = false
    cur_mtp_prev_hidden = mtp_prev_hidden
    cur_mtp_input_hidden = mtp_input_hidden
    cur_mtp_input_token = mtp_input_token
    cur_mtp_input_pos = mtp_input_pos
    rejected_index = -1
    proposal.each_with_index do |cand, i|
      accepted_before = accepted_draft_tokens
      rejections_before = rejections
      second_id = tree2_enabled ? read_second_id.call(current_block, i) : -1_i32
      margin = tree2_enabled ? read_top2_margin.call(current_block, i) : nil
      top1_hit = cand == expected
      top2_hit = second_id == expected
      rejected_now = false
      exact_ids << expected
      emitted = if top1_hit
                  accepted_draft_tokens += 1
                  cand
                else
                  if tree2_enabled && !guard_rejected && !(branch_guard_resync_ready && i == branch_guard_resync_index)
                    record_tree2_reject_margin.call(margin) if margin
                  end
                  if mtp_k2_on_reject_enabled && second_id != expected
                    mtp = mtp_k2_on_reject.not_nil!
                    mtp_out = mtp_hidden_topk_for_fusion(weights, mtp, cur_mtp_prev_hidden.not_nil!,
                      cur_mtp_input_token, cur_mtp_input_pos, 2)
                    mtp_k2_reject_ms += mtp_out[:ms]
                    mtp_k2_reject_checks += 1
                    if mtp_out[:topk].map(&.[0]).includes?(expected)
                      mtp_k2_reject_rescues += 1
                    else
                      mtp_k2_reject_misses += 1
                    end
                  end
                  draft_wasted_tail_tokens += proposal.size - i - 1
                  rejections += 1
                  rejected = true
                  rejected_index = i
                  rejected_now = true
                  expected
                end
      trace_self_spec_router_token(chunks, i, emitted_tokens, proposal.size, verifier_tokens.size,
        margin, proposal_margin_min, top1_hit, top2_hit, second_id >= 0, rejected_now,
        branch_guard_index, guard_index, risk_offramp, !next_block.nil?, current_block.use_updown,
        rejections_before, accepted_before)
      correction_or_accepted << emitted
      emitted_ids << emitted
      emitted_tokens += 1

      pos = cycle_start_pos + i
      last_token = emitted
      pos_last = pos
      if mtp_k2_on_reject_enabled && top1_hit && i < target_hiddens.size // hp.n_embd
        cur_mtp_prev_hidden = cur_mtp_input_hidden.not_nil!
        cur_mtp_input_hidden = target_hiddens[i * hp.n_embd, hp.n_embd]
        cur_mtp_input_token = emitted
        cur_mtp_input_pos = pos
      end
      expected = target_nexts[i][0] if top1_hit && i < target_nexts.size
      break if rejected || emitted_tokens >= gen_tokens
    end
    controller_ms += (Time.instant - t_controller).total_milliseconds
    wba.try(&.mark("controller", "accept_chunk_#{chunks}", t_controller, Time.instant))
    accept_history << (accepted_draft_tokens - chunk_accepted_start)
    reject_index_history << rejected_index

    if rejected
      draft_wasted_next_tokens += next_proposal_limit if next_block
      drain_block.call(next_block)
      disable_updown_after_reject.call(current_block.use_updown)
      if emitted_tokens < gen_tokens
        resync_base = nil.as(ML::GGUF::Qwen35CPU::State?)
        if branch_guard_resync_ready && rejected_index == branch_guard_resync_index
          branch_pos = cycle_start_pos + rejected_index
          resync_base = copy_owned_resync_base.call(verifier_state, branch_pos, "branch_guard_#{chunks}_#{rejected_index}")
          t_branch_advance = Time.instant
          target_next_id = ML::GGUF::Qwen35CPU.forward_top1(weights, last_token, pos_last, verifier_state)[0]
          dt_branch_advance = (Time.instant - t_branch_advance).total_milliseconds
          verifier_ms += dt_branch_advance
          if attr_collect
            verifier_prefill_ms += dt_branch_advance
            verifier_chunks += 1
            verifier_tokens_count += 1
            tree2_branch_guard_replayless_resyncs += 1
          end
          wba.try(&.mark("verifier", "branch_guard_advance_#{chunks}_#{rejected_index}", t_branch_advance, Time.instant))
        elsif branch_guard_snapshot_enabled && branch_guard_snapshot_state && branch_guard_index && branch_guard_snapshot_pos >= 0 && rejected_index > branch_guard_index.not_nil!
          snapshot = branch_guard_snapshot_state.not_nil!
          bgi = branch_guard_index.not_nil!
          tail_start = bgi + 1
          tail_size = correction_or_accepted.size - tail_start
          tail_tokens = correction_or_accepted[tail_start, tail_size]
          t_snapshot_restore = Time.instant
          copy_verifier_recurrent_state.call(verifier_state, snapshot, branch_guard_snapshot_pos)
          tree2_branch_guard_snapshot_restore_ms += (Time.instant - t_snapshot_restore).total_milliseconds
          t_suffix_replay = Time.instant
          corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, tail_tokens, branch_guard_snapshot_pos, verifier_state)
          dt_suffix_replay = (Time.instant - t_suffix_replay).total_milliseconds
          replay_ms += dt_suffix_replay
          tree2_branch_guard_suffix_replays += 1
          tree2_branch_guard_suffix_replay_tokens += tail_tokens.size
          tree2_branch_guard_suffix_replay_ms += dt_suffix_replay
          target_next_id = corrected[-1][0]
          accepted_tail_prefix = tail_tokens.size > 1 ? tail_tokens[0, tail_tokens.size - 1] : [] of Int32
          t_snapshot_resync_base = Time.instant
          resync_base = copy_owned_resync_base_from_branch_snapshot.call(verifier_state, snapshot, branch_guard_snapshot_pos, pos_last, accepted_tail_prefix, "branch_guard_suffix_#{chunks}_#{rejected_index}")
          tree2_branch_guard_snapshot_resync_base_ms += (Time.instant - t_snapshot_resync_base).total_milliseconds
          wba.try(&.mark("verifier", "branch_guard_suffix_replay_#{chunks}_#{rejected_index}", t_suffix_replay, Time.instant))
        elsif use_verifier_backup
          backup = verifier_backup.not_nil!
          copy_verifier_state.call(verifier_state, backup, cycle_start_pos)
          t_replay = Time.instant
          corrected = if mtp_k2_on_reject_enabled
                        replay = ML::GGUF::Qwen35CPU.prefill_tokens_hidden_top1s(weights, correction_or_accepted, cycle_start_pos, verifier_state)
                        local_prev_hidden = mtp_prev_hidden.not_nil!
                        local_input_hidden = mtp_input_hidden.not_nil!
                        local_input_token = mtp_input_token
                        local_input_pos = mtp_input_pos
                        correction_or_accepted.each_with_index do |tok, j|
                          local_prev_hidden = local_input_hidden
                          local_input_hidden = replay[:hidden][j * hp.n_embd, hp.n_embd]
                          local_input_token = tok
                          local_input_pos = cycle_start_pos + j
                        end
                        mtp_prev_hidden = local_prev_hidden
                        mtp_input_hidden = local_input_hidden
                        mtp_input_token = local_input_token
                        mtp_input_pos = local_input_pos
                        replay[:top1s]
                      else
                        ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, correction_or_accepted, cycle_start_pos, verifier_state)
                      end
          replay_ms += (Time.instant - t_replay).total_milliseconds
          target_next_id = corrected[-1][0]
          resync_base = copy_owned_resync_base.call(backup, cycle_start_pos, "resync_#{chunks}")
          if correction_or_accepted.size > 1
            ML::GGUF::Qwen35CPU.prefill_tokens(weights, correction_or_accepted[0, correction_or_accepted.size - 1], cycle_start_pos, resync_base)
          end
        else
          t_rebuild = Time.instant
          consumed = prompt_ids + emitted_ids
          verifier_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
          ML::GGUF::Qwen35CPU.prepare_state_metal!(verifier_state, hp)
          target_next_id = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, consumed, 0, verifier_state)[0]
          base_tokens = consumed[0, consumed.size - 1]
          resync_base = verifier_state_after_prefix(weights, base_tokens, max_seq)
          rebuild_ms += (Time.instant - t_rebuild).total_milliseconds
          wba.try(&.mark("controller", "rebuild_#{chunks}", t_rebuild, Time.instant))
        end
        next if finish_reject_offramp.call("#{chunks}_#{rejected_index}")
        t_resync = Time.instant
        current_schedule_index = 0
        draft_resyncs += 1
        current_block = submit_seed_owned.call(resync_base.not_nil!, last_token, pos_last, "self_spec_resync_#{chunks}", Math.min(schedule[current_schedule_index], gen_tokens - emitted_tokens), draft_updown_enabled)
        current_proposal = read_block.call(current_block, Math.min(schedule[current_schedule_index], gen_tokens - emitted_tokens), "resync_#{chunks}")
        dt_resync = (Time.instant - t_resync).total_milliseconds
        draft_seed_ms += dt_resync
        overlap_ms += dt_resync
        draft_resync_ms += dt_resync if attr_collect
        wba.try(&.mark("pipeline", "resync_#{chunks}", t_resync, Time.instant))
      end
    else
      if mtp_k2_on_reject_enabled
        mtp_prev_hidden = cur_mtp_prev_hidden
        mtp_input_hidden = cur_mtp_input_hidden
        mtp_input_token = cur_mtp_input_token
        mtp_input_pos = cur_mtp_input_pos
      end
      update_updown_after_accept.call(proposal_margin_min, proposal_margin_checks)
      update_noffn_after_accept.call(proposal_margin_min, proposal_margin_checks)
      if emitted_tokens < gen_tokens
        target_next_id = target_nexts[proposal.size - 1][0]
        if block = next_block
          draft_next_ms += chunk_draft_next_ms
          current_block = block
          current_proposal = next_proposal
        elsif refresh_current_draft
          t_next = Time.instant
          next_steps = Math.min(schedule[next_schedule_index], gen_tokens - emitted_tokens)
          exact_base = if use_verifier_backup
                         backup = verifier_backup.not_nil!
                         base = copy_owned_resync_base.call(backup, cycle_start_pos, "refresh_accept_#{chunks}")
                         if proposal.size > 1
                           ML::GGUF::Qwen35CPU.prefill_tokens(weights, proposal[0, proposal.size - 1], cycle_start_pos, base)
                         end
                         base
                       else
                         t_rebuild = Time.instant
                         consumed = prompt_ids + emitted_ids
                         base_tokens = consumed[0, consumed.size - 1]
                         rebuilt = verifier_state_after_prefix(weights, base_tokens, max_seq)
                         rebuild_ms += (Time.instant - t_rebuild).total_milliseconds
                         rebuilt
                       end
          current_block = submit_seed_owned.call(exact_base, last_token, pos_last, "self_spec_refresh_accept_#{chunks}", next_steps, draft_updown_enabled)
          current_proposal = read_block.call(current_block, next_steps, "refresh_accept_#{chunks}")
          dt_next = (Time.instant - t_next).total_milliseconds
          draft_next_ms += dt_next
          chunk_draft_next_ms += dt_next
        else
          t_next = Time.instant
          last_proposed_buf = current_block.submissions[proposal.size - 1].top1_id_buf.not_nil!
          next_steps = Math.min(schedule[next_schedule_index], gen_tokens - emitted_tokens)
          current_block = submit_routed_block.call(current_block.state, current_block.lr_bufs, current_block.full_current, last_proposed_buf, pos_last, "self_spec_risk_offramp_next_#{chunks}", nil, next_steps, draft_updown_enabled)
          current_proposal = read_block.call(current_block, next_steps, "risk_offramp_next_#{chunks}")
          draft_next_ms += (Time.instant - t_next).total_milliseconds
        end
        current_schedule_index = next_schedule_index
      end
    end
  end
  # Report real self-spec wall time. The phase counters above are diagnostic only:
  # they can overlap and previously missed reject replay/controller work.
  overlap_ms = (Time.instant - t_seed).total_milliseconds

  plain_state = state_before_last.fork
  t_plain = Time.instant
  plain_last_token = prompt_last_token
  plain_pos_last = prompt_pos_last
  plain_exact_ids = [] of Int32
  gen_tokens.times do
    id = ML::GGUF::Qwen35CPU.forward_top1(weights, plain_last_token, plain_pos_last, plain_state)[0]
    plain_exact_ids << id
    plain_last_token = id
    plain_pos_last += 1
  end
  plain_exact_ms = (Time.instant - t_plain).total_milliseconds
  raise "plain exact ids mismatch" unless plain_exact_ids == exact_ids
  wba.try(&.mark("pipeline", "plain_exact", t_plain, Time.instant))

  attr_collect = false
  active_draft_no_ffn = draft_no_ffn_candidate && !draft_no_ffn_dynamic_gate
  active_draft_no_ffn_layer_indices = active_draft_no_ffn ? draft_no_ffn_layer_indices : nil
  serial_state_before_last = state_before_last
  serial_verifier_state = state_before_last.fork
  serial_backup = if use_verifier_backup
                    backup = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
                    ML::GGUF::Qwen35CPU.prepare_state_metal!(backup, hp)
                    backup
                  else
                    nil
                  end
  t_serial = Time.instant
  serial_target_next_id = ML::GGUF::Qwen35CPU.forward_top1(weights, prompt_last_token, prompt_pos_last, serial_verifier_state)[0]
  serial_last_token = prompt_last_token
  serial_pos_last = prompt_pos_last
  serial_schedule_index = 0
  serial_draft_updown_available = updown_actual_rank > 0
  serial_draft_updown_chunks = 0
  serial_draft_updown_cap_open = ->{
    if max_chunks = draft_updown_max_chunks
      serial_draft_updown_chunks < max_chunks
    else
      true
    end
  }
  serial_draft_updown_enabled = serial_draft_updown_available && draft_updown_first_margin_threshold.nil? && draft_updown_after_rejects <= 0 && draft_updown_after_full_accepts <= 0 && draft_updown_min_margin.nil? && serial_draft_updown_cap_open.call
  serial_draft_updown_full_accept_streak = 0
  serial_rejections = 0
  serial_draft_noffn_chunks = 0
  serial_draft_noffn_full_accept_streak = 0
  serial_draft_noffn_cap_open = ->{
    if max_chunks = draft_no_ffn_max_chunks
      serial_draft_noffn_chunks < max_chunks
    else
      true
    end
  }
  serial_enable_noffn_candidate = ->{
    active_draft_no_ffn = draft_no_ffn
    active_draft_no_ffn_layer_indices = draft_no_ffn_layer_indices
  }
  serial_disable_noffn_candidate = ->{
    active_draft_no_ffn = false
    active_draft_no_ffn_layer_indices = nil
  }
  serial_update_updown_after_accept = ->(margin_min : Float64, margin_checks : Int32) {
    if serial_draft_updown_available
      if draft_updown_after_full_accepts > 0
        serial_draft_updown_full_accept_streak += 1
      end
      if draft_updown_after_full_accepts > 0 || !draft_updown_min_margin.nil?
        margin_ok = if threshold = draft_updown_min_margin
                      margin_checks > 0 && margin_min >= threshold.not_nil!
                    else
                      true
                    end
        streak_ok = draft_updown_after_full_accepts <= 0 || serial_draft_updown_full_accept_streak >= draft_updown_after_full_accepts
        serial_draft_updown_enabled = margin_ok && streak_ok && serial_draft_updown_cap_open.call
      end
    end
  }
  serial_update_noffn_after_accept = ->(margin_min : Float64, margin_checks : Int32) {
    if draft_no_ffn_candidate
      serial_draft_noffn_full_accept_streak += 1 if draft_no_ffn_after_full_accepts > 0
      if draft_no_ffn_dynamic_gate
        margin_ok = if threshold = draft_no_ffn_min_margin
                      margin_checks > 0 && margin_min >= threshold.not_nil!
                    else
                      true
                    end
        streak_ok = draft_no_ffn_after_full_accepts <= 0 || serial_draft_noffn_full_accept_streak >= draft_no_ffn_after_full_accepts
        if margin_ok && streak_ok && serial_draft_noffn_cap_open.call
          serial_enable_noffn_candidate.call
        else
          serial_disable_noffn_candidate.call
        end
      elsif !draft_no_ffn_max_chunks.nil? && !serial_draft_noffn_cap_open.call
        serial_disable_noffn_candidate.call
      end
    end
  }
  serial_disable_updown_after_reject = ->(rejected_updown : Bool) {
    serial_draft_updown_full_accept_streak = 0
    if rejected_updown
      serial_draft_updown_enabled = false if draft_updown_fallback_on_reject || draft_updown_after_rejects > 0 || draft_updown_after_full_accepts > 0 || !draft_updown_min_margin.nil?
    elsif draft_updown_after_rejects > 0 && serial_draft_updown_available && serial_rejections >= draft_updown_after_rejects && serial_draft_updown_cap_open.call
      serial_draft_updown_enabled = true
    elsif (draft_updown_fallback_on_reject || draft_updown_after_full_accepts > 0 || !draft_updown_min_margin.nil?) && serial_draft_updown_enabled
      serial_draft_updown_enabled = false
    end
    serial_draft_noffn_full_accept_streak = 0
    if draft_no_ffn_fallback_on_reject || draft_no_ffn_dynamic_gate
      serial_disable_noffn_candidate.call
    end
  }
  serial_current_block = submit_seed.call(serial_state_before_last, serial_last_token, serial_pos_last, "self_spec_serial_seed", Math.min(schedule[serial_schedule_index], gen_tokens), serial_draft_updown_enabled)
  serial_current_proposal = read_block.call(serial_current_block, Math.min(schedule[serial_schedule_index], gen_tokens), "serial_seed")
  if threshold = draft_updown_first_margin_threshold
    if serial_draft_updown_available && !serial_current_block.use_updown && gen_tokens > 0
      first_margin_min = Float64::INFINITY
      first_margin_checks = 0
      serial_current_proposal.each_index do |i|
        if margin = read_top2_margin.call(serial_current_block, i)
          first_margin_min = margin if margin < first_margin_min
          first_margin_checks += 1
        end
      end
      if first_margin_checks > 0 && first_margin_min <= threshold.not_nil!
        up_block = submit_seed.call(serial_state_before_last, serial_last_token, serial_pos_last, "self_spec_serial_seed_updown_first_margin", Math.min(schedule[serial_schedule_index], gen_tokens), true)
        up_proposal = read_block.call(up_block, Math.min(schedule[serial_schedule_index], gen_tokens), "serial_seed_updown_first_margin")
        drain_block.call(serial_current_block)
        serial_current_block = up_block
        serial_current_proposal = up_proposal
        serial_draft_updown_enabled = true
      else
        serial_draft_updown_enabled = false
      end
    end
  end
  if draft_updown_race_first_chunk && serial_draft_updown_available && !serial_current_block.use_updown && gen_tokens > 0
    serial_score_first_chunk = ->(candidate_proposal : Array(Int32)) {
      eval_state = serial_verifier_state.fork
      candidate_size = Math.min(candidate_proposal.size, gen_tokens)
      candidate = candidate_proposal[0, candidate_size]
      final_candidate = candidate_size >= gen_tokens
      verify_tokens = final_candidate && candidate.size > 1 ? candidate[0, candidate.size - 1] : candidate
      nexts = verify_tokens.empty? ? [] of {Int32, Float32} : ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, verify_tokens, prompt_ids.size, eval_state)
      expected = serial_target_next_id
      accepted = 0
      reject_index = -1
      candidate.each_with_index do |cand, i|
        if cand == expected
          accepted += 1
          expected = nexts[i][0] if i < nexts.size
        else
          reject_index = i
          break
        end
      end
      {accepted, reject_index, candidate_size}
    }

    low_score = serial_score_first_chunk.call(serial_current_proposal)
    up_block = submit_seed.call(serial_state_before_last, serial_last_token, serial_pos_last, "self_spec_serial_seed_updown_race", Math.min(schedule[serial_schedule_index], gen_tokens), true)
    up_proposal = read_block.call(up_block, Math.min(schedule[serial_schedule_index], gen_tokens), "serial_seed_updown_race")
    up_score = serial_score_first_chunk.call(up_proposal)
    low_full = low_score[0] >= low_score[2] && low_score[1] < 0
    up_full = up_score[0] >= up_score[2] && up_score[1] < 0
    if up_full && !low_full
      drain_block.call(serial_current_block)
      serial_current_block = up_block
      serial_current_proposal = up_proposal
      serial_draft_updown_enabled = true
    else
      drain_block.call(up_block)
      serial_draft_updown_enabled = false
    end
  end
  serial_emitted_tokens = 0
  serial_exact_ids = [] of Int32
  serial_emitted_ids = [] of Int32
  serial_chunks = 0

  while serial_emitted_tokens < gen_tokens
    serial_chunks += 1
    chunk_size = Math.min(serial_current_proposal.size, gen_tokens - serial_emitted_tokens)
    proposal = serial_current_proposal[0, chunk_size]
    serial_draft_updown_chunks += 1 if serial_current_block.use_updown
    serial_draft_noffn_chunks += 1 if serial_current_block.use_noffn
    serial_draft_updown_enabled = false unless serial_draft_updown_cap_open.call
    serial_disable_noffn_candidate.call unless serial_draft_noffn_cap_open.call
    cycle_start_pos = prompt_ids.size + serial_emitted_tokens
    final_chunk = serial_emitted_tokens + proposal.size >= gen_tokens
    verifier_tokens = final_chunk && proposal.size > 1 ? proposal[0, proposal.size - 1] : proposal
    serial_proposal_margin_min = Float64::INFINITY
    serial_proposal_margin_checks = 0
    if tree2_enabled
      proposal.each_index do |i|
        if margin = read_top2_margin.call(serial_current_block, i)
          serial_proposal_margin_min = margin if margin < serial_proposal_margin_min
          serial_proposal_margin_checks += 1
        end
      end
    end

    if tree2_staged_tokens > 0 && !proposal.empty?
      chunk_emitted_start = serial_emitted_tokens
      stage_offset = 0
      expected = serial_target_next_id
      rejected = false
      while stage_offset < proposal.size && serial_emitted_tokens < gen_tokens
        stage_size = Math.min(tree2_staged_tokens, proposal.size - stage_offset)
        stage_pos = cycle_start_pos + stage_offset
        stage_final_token = chunk_emitted_start + stage_offset + stage_size >= gen_tokens
        stage_verify_size = stage_final_token ? Math.max(stage_size - 1, 0) : stage_size

        if backup = serial_backup
          copy_verifier_state.call(backup, serial_verifier_state, stage_pos)
        end
        stage_verify_tokens = stage_verify_size > 0 ? proposal[stage_offset, stage_verify_size] : [] of Int32
        target_nexts = stage_verify_tokens.empty? ? [] of {Int32, Float32} : ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, stage_verify_tokens, stage_pos, serial_verifier_state)

        correction_or_accepted = [] of Int32
        stage_rejected = false
        stage_size.times do |j|
          cand = proposal[stage_offset + j]
          serial_exact_ids << expected
          emitted = if cand == expected
                      cand
                    else
                      rejected = true
                      stage_rejected = true
                      serial_rejections += 1
                      expected
                    end
          correction_or_accepted << emitted
          serial_emitted_ids << emitted
          serial_emitted_tokens += 1
          serial_last_token = emitted
          serial_pos_last = stage_pos + j
          expected = target_nexts[j][0] if cand == emitted && j < target_nexts.size
          break if stage_rejected || serial_emitted_tokens >= gen_tokens
        end

        if stage_rejected
          serial_disable_updown_after_reject.call(serial_current_block.use_updown)
          serial_resync_base = nil.as(ML::GGUF::Qwen35CPU::State?)
          if serial_emitted_tokens < gen_tokens
            if use_verifier_backup
              backup = serial_backup.not_nil!
              copy_verifier_state.call(serial_verifier_state, backup, stage_pos)
              corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, correction_or_accepted, stage_pos, serial_verifier_state)
              serial_target_next_id = corrected[-1][0]
              serial_resync_base = copy_owned_resync_base.call(backup, stage_pos, "serial_staged_#{serial_chunks}_#{stage_offset}")
              if correction_or_accepted.size > 1
                ML::GGUF::Qwen35CPU.prefill_tokens(weights, correction_or_accepted[0, correction_or_accepted.size - 1], stage_pos, serial_resync_base)
              end
            else
              consumed = prompt_ids + serial_emitted_ids
              serial_verifier_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
              ML::GGUF::Qwen35CPU.prepare_state_metal!(serial_verifier_state, hp)
              serial_target_next_id = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, consumed, 0, serial_verifier_state)[0]
              base_tokens = consumed[0, consumed.size - 1]
              serial_resync_base = verifier_state_after_prefix(weights, base_tokens, max_seq)
            end

            serial_schedule_index = 0
            serial_current_block = submit_seed_owned.call(serial_resync_base.not_nil!, serial_last_token, serial_pos_last, "self_spec_serial_staged_resync_#{serial_chunks}", Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), serial_draft_updown_enabled)
            serial_current_proposal = read_block.call(serial_current_block, Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), "serial_staged_resync_#{serial_chunks}")
          end
          break
        end

        stage_offset += stage_size
      end

      unless rejected
        serial_update_updown_after_accept.call(serial_proposal_margin_min, serial_proposal_margin_checks)
        serial_update_noffn_after_accept.call(serial_proposal_margin_min, serial_proposal_margin_checks)
        if serial_emitted_tokens < gen_tokens
          serial_target_next_id = expected
          last_proposed_buf = serial_current_block.submissions[proposal.size - 1].top1_id_buf.not_nil!
          serial_schedule_index = (serial_schedule_index + 1) % schedule.size
          serial_current_block = submit_routed_block.call(serial_current_block.state, serial_current_block.lr_bufs, serial_current_block.full_current, last_proposed_buf, serial_pos_last, "self_spec_serial_staged_next_#{serial_chunks}", nil, Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), serial_draft_updown_enabled)
          serial_current_proposal = read_block.call(serial_current_block, Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), "serial_staged_next_#{serial_chunks}")
        end
      end
      next
    end

    if tree2_anywhere && !proposal.empty?
      rejected = false
      proposal.each_with_index do |cand, i|
        expected = serial_target_next_id
        serial_exact_ids << expected
        emitted = if cand == expected
                    cand
                  else
                    rejected = true
                    serial_rejections += 1
                    expected
                  end
        serial_emitted_ids << emitted
        serial_emitted_tokens += 1
        pos = cycle_start_pos + i
        serial_last_token = emitted
        serial_pos_last = pos

        if serial_emitted_tokens < gen_tokens
          serial_resync_base = rejected ? copy_owned_resync_base.call(serial_verifier_state, pos, "serial_tree2_anywhere_#{serial_chunks}_#{i}") : nil
          serial_target_next_id = ML::GGUF::Qwen35CPU.forward_top1(weights, emitted, pos, serial_verifier_state)[0]

          if rejected
            serial_disable_updown_after_reject.call(serial_current_block.use_updown)
            serial_schedule_index = 0
            serial_current_block = submit_seed_owned.call(serial_resync_base.not_nil!, serial_last_token, serial_pos_last, "self_spec_serial_tree2_anywhere_#{serial_chunks}", Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), serial_draft_updown_enabled)
            serial_current_proposal = read_block.call(serial_current_block, Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), "serial_tree2_anywhere_#{serial_chunks}")
          end
        end
        break if rejected || serial_emitted_tokens >= gen_tokens
      end

      unless rejected
        serial_update_updown_after_accept.call(serial_proposal_margin_min, serial_proposal_margin_checks)
        serial_update_noffn_after_accept.call(serial_proposal_margin_min, serial_proposal_margin_checks)
        if serial_emitted_tokens < gen_tokens
          serial_schedule_index = (serial_schedule_index + 1) % schedule.size
          last_proposed_buf = serial_current_block.submissions[proposal.size - 1].top1_id_buf.not_nil!
          serial_current_block = submit_routed_block.call(serial_current_block.state, serial_current_block.lr_bufs, serial_current_block.full_current, last_proposed_buf, serial_pos_last, "self_spec_serial_tree2_anywhere_next_#{serial_chunks}", nil, Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), serial_draft_updown_enabled)
          serial_current_proposal = read_block.call(serial_current_block, Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), "serial_tree2_anywhere_next_#{serial_chunks}")
        end
      end
      next
    end

    if tree2_first && !proposal.empty? && proposal[0] != serial_target_next_id
      expected = serial_target_next_id
      serial_exact_ids << expected
      serial_emitted_ids << expected
      serial_emitted_tokens += 1
      serial_last_token = expected
      serial_pos_last = cycle_start_pos
      serial_rejections += 1
      serial_disable_updown_after_reject.call(serial_current_block.use_updown)

      if serial_emitted_tokens < gen_tokens
        serial_resync_base = copy_owned_resync_base.call(serial_verifier_state, cycle_start_pos, "serial_tree2_first_#{serial_chunks}")
        serial_target_next_id = ML::GGUF::Qwen35CPU.forward_top1(weights, expected, cycle_start_pos, serial_verifier_state)[0]
        if reject_offramp_after > 0 && serial_rejections >= reject_offramp_after
          while serial_emitted_tokens < gen_tokens
            serial_expected = serial_target_next_id
            serial_exact_ids << serial_expected
            serial_emitted_ids << serial_expected
            serial_last_token = serial_expected
            serial_pos_last = prompt_ids.size + serial_emitted_tokens
            serial_emitted_tokens += 1
            if serial_emitted_tokens < gen_tokens
              serial_target_next_id = ML::GGUF::Qwen35CPU.forward_top1(weights, serial_last_token, serial_pos_last, serial_verifier_state)[0]
            end
          end
          next
        end
        serial_schedule_index = 0
        serial_current_block = submit_seed_owned.call(serial_resync_base, serial_last_token, serial_pos_last, "self_spec_serial_tree2_first_#{serial_chunks}", Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), serial_draft_updown_enabled)
        serial_current_proposal = read_block.call(serial_current_block, Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), "serial_tree2_first_#{serial_chunks}")
      end
      next
    end

    serial_branch_guard_index = nil.as(Int32?)
    if branch_threshold = tree2_branch_guard
      if !branch_guard_until_reject || serial_rejections == 0
        verifier_tokens.each_index do |i|
          if margin = read_top2_margin.call(serial_current_block, i)
            if margin <= branch_threshold
              if branch_guard_split_allowed.call(serial_current_block, i, verifier_tokens.size)
                serial_branch_guard_index = i
                break
              end
            end
          end
        end
      end
    end
    if backup = serial_backup
      copy_verifier_state.call(backup, serial_verifier_state, cycle_start_pos)
    end
    serial_guard_index = nil.as(Int32?)
    if guard_threshold = tree2_margin_guard
      verifier_tokens.each_index do |i|
        if margin = read_top2_margin.call(serial_current_block, i)
          if margin <= guard_threshold
            serial_guard_index = i
            break
          end
        end
      end
    end
    serial_guard_rejected = false
    serial_branch_guard_resync_ready = false
    serial_branch_guard_resync_index = -1
    serial_branch_guard_snapshot_state = nil.as(ML::GGUF::Qwen35CPU::State?)
    serial_branch_guard_snapshot_pos = -1
    target_nexts = [] of {Int32, Float32}
    if bgi = serial_branch_guard_index
      prefix_size = bgi
      prefix_ok = true
      expected_branch = serial_target_next_id
      if prefix_size > 0
        prefix_tokens = proposal[0, prefix_size]
        target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, prefix_tokens, cycle_start_pos, serial_verifier_state)
        prefix_size.times do |i|
          cand = proposal[i]
          if cand == expected_branch
            expected_branch = target_nexts[i][0]
          else
            prefix_ok = false
            break
          end
        end
      end
      if prefix_ok
        guard_expected = expected_branch
        guard_cand = proposal[bgi]
        if guard_cand == guard_expected
          if branch_guard_snapshot_enabled
            guard_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, [guard_cand], cycle_start_pos + bgi, serial_verifier_state)
            target_nexts.concat(guard_nexts)
            suffix_size = verifier_tokens.size - bgi - 1
            if suffix_size > 0 && bgi + 1 >= branch_guard_snapshot_min_prefix && branch_guard_snapshot_suffix_allowed.call(serial_current_block, bgi + 1, verifier_tokens.size)
              snapshot = branch_guard_snapshot_scratch.not_nil!
              serial_branch_guard_snapshot_pos = cycle_start_pos + bgi + 1
              copy_verifier_recurrent_state.call(snapshot, serial_verifier_state, serial_branch_guard_snapshot_pos)
              serial_branch_guard_snapshot_state = snapshot
              suffix_tokens = verifier_tokens[bgi + 1, suffix_size]
              suffix_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, suffix_tokens, cycle_start_pos + bgi + 1, serial_verifier_state)
              target_nexts.concat(suffix_nexts)
            elsif suffix_size > 0
              suffix_tokens = verifier_tokens[bgi + 1, suffix_size]
              suffix_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, suffix_tokens, cycle_start_pos + bgi + 1, serial_verifier_state)
              target_nexts.concat(suffix_nexts)
            end
          else
            suffix_size = verifier_tokens.size - bgi
            if suffix_size > 0
              suffix_tokens = verifier_tokens[bgi, suffix_size]
              suffix_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, suffix_tokens, cycle_start_pos + bgi, serial_verifier_state)
              target_nexts.concat(suffix_nexts)
            end
          end
        else
          serial_branch_guard_resync_ready = true
          serial_branch_guard_resync_index = bgi
        end
      end
    elsif gi = serial_guard_index
      guard_verify_size = gi + 1
      guard_tokens = proposal[0, guard_verify_size]
      target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, guard_tokens, cycle_start_pos, serial_verifier_state)
      expected_guard = serial_target_next_id
      guard_verify_size.times do |i|
        cand = proposal[i]
        if cand == expected_guard
          expected_guard = target_nexts[i][0]
        else
          serial_guard_rejected = true
          break
        end
      end
      unless serial_guard_rejected
        suffix_size = verifier_tokens.size - guard_verify_size
        if suffix_size > 0
          suffix_tokens = verifier_tokens[guard_verify_size, suffix_size]
          suffix_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, suffix_tokens, cycle_start_pos + guard_verify_size, serial_verifier_state)
          target_nexts.concat(suffix_nexts)
        end
      end
    else
      target_nexts = verifier_tokens.empty? ? [] of {Int32, Float32} : ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, verifier_tokens, cycle_start_pos, serial_verifier_state)
    end

    correction_or_accepted = [] of Int32
    expected = serial_target_next_id
    rejected = false
    serial_rejected_index = -1
    proposal.each_with_index do |cand, i|
      serial_exact_ids << expected
      emitted = if cand == expected
                  cand
                else
                  rejected = true
                  serial_rejected_index = i
                  serial_rejections += 1
                  expected
                end
      correction_or_accepted << emitted
      serial_emitted_ids << emitted
      serial_emitted_tokens += 1
      serial_last_token = emitted
      serial_pos_last = cycle_start_pos + i
      expected = target_nexts[i][0] if cand == expected && i < target_nexts.size
      break if rejected || serial_emitted_tokens >= gen_tokens
    end

    if rejected
      serial_disable_updown_after_reject.call(serial_current_block.use_updown)
      if serial_emitted_tokens < gen_tokens
        serial_resync_base = nil.as(ML::GGUF::Qwen35CPU::State?)
        if serial_branch_guard_resync_ready && serial_rejected_index == serial_branch_guard_resync_index
          branch_pos = cycle_start_pos + serial_rejected_index
          serial_resync_base = copy_owned_resync_base.call(serial_verifier_state, branch_pos, "serial_branch_guard_#{serial_chunks}_#{serial_rejected_index}")
          serial_target_next_id = ML::GGUF::Qwen35CPU.forward_top1(weights, serial_last_token, serial_pos_last, serial_verifier_state)[0]
        elsif branch_guard_snapshot_enabled && serial_branch_guard_snapshot_state && serial_branch_guard_index && serial_branch_guard_snapshot_pos >= 0 && serial_rejected_index > serial_branch_guard_index.not_nil!
          snapshot = serial_branch_guard_snapshot_state.not_nil!
          bgi = serial_branch_guard_index.not_nil!
          tail_start = bgi + 1
          tail_size = correction_or_accepted.size - tail_start
          tail_tokens = correction_or_accepted[tail_start, tail_size]
          copy_verifier_recurrent_state.call(serial_verifier_state, snapshot, serial_branch_guard_snapshot_pos)
          corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, tail_tokens, serial_branch_guard_snapshot_pos, serial_verifier_state)
          serial_target_next_id = corrected[-1][0]
          accepted_tail_prefix = tail_tokens.size > 1 ? tail_tokens[0, tail_tokens.size - 1] : [] of Int32
          serial_resync_base = copy_owned_resync_base_from_branch_snapshot.call(serial_verifier_state, snapshot, serial_branch_guard_snapshot_pos, serial_pos_last, accepted_tail_prefix, "serial_branch_guard_suffix_#{serial_chunks}_#{serial_rejected_index}")
        elsif use_verifier_backup
          backup = serial_backup.not_nil!
          copy_verifier_state.call(serial_verifier_state, backup, cycle_start_pos)
          corrected = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, correction_or_accepted, cycle_start_pos, serial_verifier_state)
          serial_target_next_id = corrected[-1][0]
          serial_resync_base = copy_owned_resync_base.call(backup, cycle_start_pos, "serial_resync_#{serial_chunks}")
          if correction_or_accepted.size > 1
            ML::GGUF::Qwen35CPU.prefill_tokens(weights, correction_or_accepted[0, correction_or_accepted.size - 1], cycle_start_pos, serial_resync_base)
          end
        else
          consumed = prompt_ids + serial_emitted_ids
          serial_verifier_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
          ML::GGUF::Qwen35CPU.prepare_state_metal!(serial_verifier_state, hp)
          serial_target_next_id = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, consumed, 0, serial_verifier_state)[0]
          base_tokens = consumed[0, consumed.size - 1]
          serial_resync_base = verifier_state_after_prefix(weights, base_tokens, max_seq)
        end
        serial_schedule_index = 0
        serial_current_block = submit_seed_owned.call(serial_resync_base.not_nil!, serial_last_token, serial_pos_last, "self_spec_serial_resync_#{serial_chunks}", Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), serial_draft_updown_enabled)
        serial_current_proposal = read_block.call(serial_current_block, Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens), "serial_resync_#{serial_chunks}")
      end
    else
      serial_refresh_current_draft = draft_refresh_on_accept || (draft_updown_refresh_on_accept && serial_current_block.use_updown)
      serial_update_updown_after_accept.call(serial_proposal_margin_min, serial_proposal_margin_checks)
      serial_update_noffn_after_accept.call(serial_proposal_margin_min, serial_proposal_margin_checks)
      if serial_emitted_tokens < gen_tokens
        serial_target_next_id = target_nexts[proposal.size - 1][0]
        serial_schedule_index = (serial_schedule_index + 1) % schedule.size
        serial_next_steps = Math.min(schedule[serial_schedule_index], gen_tokens - serial_emitted_tokens)
        if serial_refresh_current_draft
          exact_base = if use_verifier_backup
                         backup = serial_backup.not_nil!
                         base = copy_owned_resync_base.call(backup, cycle_start_pos, "serial_refresh_accept_#{serial_chunks}")
                         if proposal.size > 1
                           ML::GGUF::Qwen35CPU.prefill_tokens(weights, proposal[0, proposal.size - 1], cycle_start_pos, base)
                         end
                         base
                       else
                         consumed = prompt_ids + serial_emitted_ids
                         base_tokens = consumed[0, consumed.size - 1]
                         verifier_state_after_prefix(weights, base_tokens, max_seq)
                       end
          serial_current_block = submit_seed_owned.call(exact_base, serial_last_token, serial_pos_last, "self_spec_serial_refresh_accept_#{serial_chunks}", serial_next_steps, serial_draft_updown_enabled)
          serial_current_proposal = read_block.call(serial_current_block, serial_next_steps, "serial_refresh_accept_#{serial_chunks}")
        else
          last_proposed_buf = serial_current_block.submissions[proposal.size - 1].top1_id_buf.not_nil!
          serial_current_block = submit_routed_block.call(serial_current_block.state, serial_current_block.lr_bufs, serial_current_block.full_current, last_proposed_buf, serial_pos_last, "self_spec_serial_next_#{serial_chunks}", nil, serial_next_steps, serial_draft_updown_enabled)
          serial_current_proposal = read_block.call(serial_current_block, serial_next_steps, "serial_next_#{serial_chunks}")
        end
      end
    end
  end
  serial_ms = (Time.instant - t_serial).total_milliseconds
  raise "serial pipeline exact ids mismatch" unless serial_exact_ids == exact_ids
  raise "serial pipeline emitted ids mismatch" unless serial_emitted_ids == emitted_ids
  wba.try(&.mark("pipeline", "paired_serial", t_serial, Time.instant))
  wba.try(&.flush)
  hidden_ms = serial_ms - overlap_ms
  agreement_margin_sweep = agreement_margin_thresholds.map_with_index do |threshold, i|
    "#{threshold}:#{agreement_margin_selected[i]}/#{agreement_margin_selected_passes[i]}/#{agreement_margin_selected_fails[i]}/#{agreement_margin_false_negatives[i]}"
  end.join(";")
  {
    chunks:                       chunks,
    rejections:                   rejections,
    accepted_draft_tokens:        accepted_draft_tokens,
    proposed_tokens:              proposed_tokens,
    draft_updown_chunks:          draft_updown_chunks,
    draft_noffn_chunks:          draft_noffn_chunks,
    draft_updown_agreement_checks: draft_updown_agreement_checks,
    draft_updown_agreement_passes: draft_updown_agreement_top1 + draft_updown_agreement_top2,
    draft_updown_agreement_top1:   draft_updown_agreement_top1,
    draft_updown_agreement_top2:   draft_updown_agreement_top2,
    draft_updown_agreement_fails:  draft_updown_agreement_fails,
    draft_updown_agreement_probe_ms: draft_updown_agreement_probe_ms,
    draft_updown_agreement_margin_min_avg: agreement_margin_count > 0 ? agreement_margin_sum / agreement_margin_count : 0.0,
    draft_updown_agreement_margin_pass_avg: agreement_margin_pass_count > 0 ? agreement_margin_pass_sum / agreement_margin_pass_count : 0.0,
    draft_updown_agreement_margin_fail_avg: agreement_margin_fail_count > 0 ? agreement_margin_fail_sum / agreement_margin_fail_count : 0.0,
    draft_updown_agreement_margin_sweep: agreement_margin_sweep,
    tree2_first_checks:           tree2_first_checks,
    tree2_first_rescues:          tree2_first_rescues,
    tree2_first_misses:           tree2_first_misses,
    tree2_first_early_exits:      tree2_first_early_exits,
    tree2_anywhere_checks:        tree2_anywhere_checks,
    tree2_anywhere_rescues:       tree2_anywhere_rescues,
    tree2_anywhere_misses:        tree2_anywhere_misses,
    tree2_anywhere_early_exits:   tree2_anywhere_early_exits,
    tree2_staged_checks:          tree2_staged_checks,
    tree2_staged_rescues:         tree2_staged_rescues,
    tree2_staged_misses:          tree2_staged_misses,
    tree2_staged_early_exits:     tree2_staged_early_exits,
    tree2_staged_stages:          tree2_staged_stages,
    tree2_margin_checks:          tree2_margin_checks,
    tree2_margin_avg:             tree2_margin_checks > 0 ? tree2_margin_sum / tree2_margin_checks : 0.0,
    tree2_margin_min:             tree2_margin_checks > 0 ? tree2_margin_min : 0.0,
    tree2_reject_margin_checks:   tree2_reject_margin_checks,
    tree2_reject_margin_avg:      tree2_reject_margin_checks > 0 ? tree2_reject_margin_sum / tree2_reject_margin_checks : 0.0,
    tree2_reject_margin_min:      tree2_reject_margin_checks > 0 ? tree2_reject_margin_min : 0.0,
    tree2_margin_guard_threshold: tree2_margin_guard || 0.0,
    tree2_margin_guard_hits:      tree2_margin_guard_hits,
    tree2_margin_guard_tokens:    tree2_margin_guard_tokens,
    tree2_margin_guard_rejects:   tree2_margin_guard_rejects,
    tree2_margin_guard_passes:    tree2_margin_guard_passes,
    tree2_branch_guard_threshold: tree2_branch_guard || 0.0,
    tree2_branch_guard_hits:      tree2_branch_guard_hits,
    tree2_branch_guard_tokens:    tree2_branch_guard_tokens,
    tree2_branch_guard_rejects:   tree2_branch_guard_rejects,
    tree2_branch_guard_rescues:   tree2_branch_guard_rescues,
    tree2_branch_guard_misses:    tree2_branch_guard_misses,
    tree2_branch_guard_passes:    tree2_branch_guard_passes,
    tree2_branch_guard_prefix_rejects: tree2_branch_guard_prefix_rejects,
    tree2_branch_guard_replayless_resyncs: tree2_branch_guard_replayless_resyncs,
    tree2_branch_guard_snapshot_copies: tree2_branch_guard_snapshot_copies,
    tree2_branch_guard_snapshot_ms: tree2_branch_guard_snapshot_ms,
    tree2_branch_guard_snapshot_restore_ms: tree2_branch_guard_snapshot_restore_ms,
    tree2_branch_guard_snapshot_resync_base_ms: tree2_branch_guard_snapshot_resync_base_ms,
    tree2_branch_guard_suffix_replays: tree2_branch_guard_suffix_replays,
    tree2_branch_guard_suffix_replay_tokens: tree2_branch_guard_suffix_replay_tokens,
    tree2_branch_guard_suffix_replay_ms: tree2_branch_guard_suffix_replay_ms,
    tree2_branch_guard_prefix_verify_ms: tree2_branch_guard_prefix_verify_ms,
    tree2_branch_guard_prefix_verify_tokens: tree2_branch_guard_prefix_verify_tokens,
    tree2_branch_guard_token_verify_ms: tree2_branch_guard_token_verify_ms,
    tree2_branch_guard_token_verify_tokens: tree2_branch_guard_token_verify_tokens,
    tree2_branch_guard_suffix_verify_ms: tree2_branch_guard_suffix_verify_ms,
    tree2_branch_guard_suffix_verify_tokens: tree2_branch_guard_suffix_verify_tokens,
    tree2_branch_guard_snapshot_suffix_verify_ms: tree2_branch_guard_snapshot_suffix_verify_ms,
    tree2_branch_guard_snapshot_suffix_verify_tokens: tree2_branch_guard_snapshot_suffix_verify_tokens,
    tree2_branch_guard_no_snapshot_suffix_verify_ms: tree2_branch_guard_no_snapshot_suffix_verify_ms,
    tree2_branch_guard_no_snapshot_suffix_verify_tokens: tree2_branch_guard_no_snapshot_suffix_verify_tokens,
    risk_offramp_threshold:       risk_offramp_margin || 0.0,
    risk_offramp_hits:            risk_offramp_hits,
    risk_offramp_delayed_blocks:  risk_offramp_delayed_blocks,
    risk_offramp_delayed_tokens:  risk_offramp_delayed_tokens,
    mtp_k2_reject_checks:         mtp_k2_reject_checks,
    mtp_k2_reject_rescues:        mtp_k2_reject_rescues,
    mtp_k2_reject_misses:         mtp_k2_reject_misses,
    mtp_k2_reject_ms:             mtp_k2_reject_ms,
    reject_offramp_after:         reject_offramp_after,
    reject_offramp_hits:          reject_offramp_hits,
    reject_offramp_tokens:        reject_offramp_tokens,
    reject_offramp_ms:            reject_offramp_ms,
    draft_seed_ms:                draft_seed_ms,
    draft_next_ms:                draft_next_ms,
    verifier_ms:                  verifier_ms,
    draft_wait_ms:                draft_wait_ms,
    backup_ms:                    backup_ms,
    rebuild_ms:                   rebuild_ms,
    controller_ms:                controller_ms,
    plain_exact_ms:               plain_exact_ms,
    serial_ms:                    serial_ms,
    overlap_ms:                   overlap_ms,
    replay_ms:                    replay_ms,
    hidden_ms:                    hidden_ms,
    speedup:                      overlap_ms > 0.0 ? serial_ms / overlap_ms : 0.0,
    plain_speedup:                overlap_ms > 0.0 ? plain_exact_ms / overlap_ms : 0.0,
    parity:                       exact_ids == emitted_ids,
    gamma_history:                gamma_history,
    accept_history:               accept_history,
    reject_index_history:         reject_index_history,
    exact_ids:                    exact_ids,
    emitted_ids:                  emitted_ids,
    draft_steps:                  draft_steps,
    draft_blocks:                 draft_blocks,
    draft_fork_ms:                draft_fork_ms,
    draft_token_buf_ms:           draft_token_buf_ms,
    draft_lr_project_ms:          draft_lr_project_ms,
    draft_submit_ms:              draft_submit_ms,
    draft_commit_ms:              draft_commit_ms,
    draft_wait_block_ms:          draft_wait_block_ms,
    draft_read_ids_ms:            draft_read_ids_ms,
    draft_resync_ms:              draft_resync_ms,
    draft_resyncs:                draft_resyncs,
    draft_wasted_tail_tokens:     draft_wasted_tail_tokens,
    draft_wasted_next_tokens:     draft_wasted_next_tokens,
    verifier_initial_ms:          verifier_initial_ms,
    verifier_prefill_ms:          verifier_prefill_ms,
    verifier_chunks:              verifier_chunks,
    verifier_tokens:              verifier_tokens_count,
    verifier_tail_skip_tokens:    verifier_tail_skip_tokens,
  }
end

private def self_spec_pipeline_attr_note(pipe) : String
  draft_profiled_ms = pipe[:draft_fork_ms] + pipe[:draft_token_buf_ms] + pipe[:draft_lr_project_ms] +
                      pipe[:draft_submit_ms] + pipe[:draft_commit_ms] + pipe[:draft_wait_block_ms] +
                      pipe[:draft_read_ids_ms]
  verifier_total_ms = pipe[:verifier_initial_ms] + pipe[:verifier_prefill_ms]
  sprintf(" attr_draft_steps=%d attr_draft_blocks=%d attr_draft_profiled_ms=%.3f attr_draft_fork_ms=%.3f attr_draft_token_buf_ms=%.3f attr_draft_lr_project_ms=%.3f attr_draft_submit_ms=%.3f attr_draft_commit_ms=%.3f attr_draft_wait_block_ms=%.3f attr_draft_read_ids_ms=%.3f attr_draft_resync_ms=%.3f attr_draft_resyncs=%d attr_draft_wasted_tail_tokens=%d attr_draft_wasted_next_tokens=%d attr_verifier_total_ms=%.3f attr_verifier_initial_ms=%.3f attr_verifier_prefill_ms=%.3f attr_verifier_chunks=%d attr_verifier_tokens=%d attr_verifier_tail_skip_tokens=%d",
    pipe[:draft_steps], pipe[:draft_blocks],
    draft_profiled_ms,
    pipe[:draft_fork_ms], pipe[:draft_token_buf_ms], pipe[:draft_lr_project_ms],
    pipe[:draft_submit_ms], pipe[:draft_commit_ms], pipe[:draft_wait_block_ms],
    pipe[:draft_read_ids_ms], pipe[:draft_resync_ms],
    pipe[:draft_resyncs], pipe[:draft_wasted_tail_tokens], pipe[:draft_wasted_next_tokens],
    verifier_total_ms, pipe[:verifier_initial_ms], pipe[:verifier_prefill_ms],
    pipe[:verifier_chunks], pipe[:verifier_tokens], pipe[:verifier_tail_skip_tokens])
end

private def self_spec_pipeline_updown_agreement_note(pipe) : String
  sweep_note = pipe[:draft_updown_agreement_margin_sweep].empty? ? "" : " draft_pca_updown_agreement_margin_sweep=#{pipe[:draft_updown_agreement_margin_sweep]}"
  sprintf(" draft_pca_updown_agreement_checks=%d draft_pca_updown_agreement_passes=%d draft_pca_updown_agreement_top1=%d draft_pca_updown_agreement_top2=%d draft_pca_updown_agreement_fails=%d draft_pca_updown_agreement_probe_ms=%.3f draft_pca_updown_agreement_margin_min_avg=%.4f draft_pca_updown_agreement_margin_pass_avg=%.4f draft_pca_updown_agreement_margin_fail_avg=%.4f%s",
    pipe[:draft_updown_agreement_checks],
    pipe[:draft_updown_agreement_passes],
    pipe[:draft_updown_agreement_top1],
    pipe[:draft_updown_agreement_top2],
    pipe[:draft_updown_agreement_fails],
    pipe[:draft_updown_agreement_probe_ms],
    pipe[:draft_updown_agreement_margin_min_avg],
    pipe[:draft_updown_agreement_margin_pass_avg],
    pipe[:draft_updown_agreement_margin_fail_avg],
    sweep_note)
end

private def self_spec_pipeline_tree2_note(pipe) : String
  sprintf(" tree2_first_checks=%d tree2_first_rescues=%d tree2_first_misses=%d tree2_first_early_exits=%d tree2_anywhere_checks=%d tree2_anywhere_rescues=%d tree2_anywhere_misses=%d tree2_anywhere_early_exits=%d tree2_staged_checks=%d tree2_staged_rescues=%d tree2_staged_misses=%d tree2_staged_early_exits=%d tree2_staged_stages=%d tree2_margin_checks=%d tree2_margin_avg=%.4f tree2_margin_min=%.4f tree2_reject_margin_checks=%d tree2_reject_margin_avg=%.4f tree2_reject_margin_min=%.4f tree2_margin_guard_threshold=%.4f tree2_margin_guard_hits=%d tree2_margin_guard_tokens=%d tree2_margin_guard_rejects=%d tree2_margin_guard_passes=%d tree2_branch_guard_threshold=%.4f tree2_branch_guard_hits=%d tree2_branch_guard_tokens=%d tree2_branch_guard_rejects=%d tree2_branch_guard_rescues=%d tree2_branch_guard_misses=%d tree2_branch_guard_passes=%d tree2_branch_guard_prefix_rejects=%d tree2_branch_guard_replayless_resyncs=%d tree2_branch_guard_snapshot_copies=%d tree2_branch_guard_snapshot_ms=%.3f tree2_branch_guard_snapshot_restore_ms=%.3f tree2_branch_guard_snapshot_resync_base_ms=%.3f tree2_branch_guard_suffix_replays=%d tree2_branch_guard_suffix_replay_tokens=%d tree2_branch_guard_suffix_replay_ms=%.3f tree2_branch_guard_prefix_verify_ms=%.3f tree2_branch_guard_prefix_verify_tokens=%d tree2_branch_guard_token_verify_ms=%.3f tree2_branch_guard_token_verify_tokens=%d tree2_branch_guard_suffix_verify_ms=%.3f tree2_branch_guard_suffix_verify_tokens=%d tree2_branch_guard_snapshot_suffix_verify_ms=%.3f tree2_branch_guard_snapshot_suffix_verify_tokens=%d tree2_branch_guard_no_snapshot_suffix_verify_ms=%.3f tree2_branch_guard_no_snapshot_suffix_verify_tokens=%d risk_offramp_threshold=%.4f risk_offramp_hits=%d risk_offramp_delayed_blocks=%d risk_offramp_delayed_tokens=%d mtp_k2_reject_checks=%d mtp_k2_reject_rescues=%d mtp_k2_reject_misses=%d mtp_k2_reject_ms=%.3f reject_offramp_after=%d reject_offramp_hits=%d reject_offramp_tokens=%d reject_offramp_ms=%.3f",
    pipe[:tree2_first_checks],
    pipe[:tree2_first_rescues],
    pipe[:tree2_first_misses],
    pipe[:tree2_first_early_exits],
    pipe[:tree2_anywhere_checks],
    pipe[:tree2_anywhere_rescues],
    pipe[:tree2_anywhere_misses],
    pipe[:tree2_anywhere_early_exits],
    pipe[:tree2_staged_checks],
    pipe[:tree2_staged_rescues],
    pipe[:tree2_staged_misses],
    pipe[:tree2_staged_early_exits],
    pipe[:tree2_staged_stages],
    pipe[:tree2_margin_checks],
    pipe[:tree2_margin_avg],
    pipe[:tree2_margin_min],
    pipe[:tree2_reject_margin_checks],
    pipe[:tree2_reject_margin_avg],
    pipe[:tree2_reject_margin_min],
    pipe[:tree2_margin_guard_threshold],
    pipe[:tree2_margin_guard_hits],
    pipe[:tree2_margin_guard_tokens],
    pipe[:tree2_margin_guard_rejects],
    pipe[:tree2_margin_guard_passes],
    pipe[:tree2_branch_guard_threshold],
    pipe[:tree2_branch_guard_hits],
    pipe[:tree2_branch_guard_tokens],
    pipe[:tree2_branch_guard_rejects],
    pipe[:tree2_branch_guard_rescues],
    pipe[:tree2_branch_guard_misses],
    pipe[:tree2_branch_guard_passes],
    pipe[:tree2_branch_guard_prefix_rejects],
    pipe[:tree2_branch_guard_replayless_resyncs],
    pipe[:tree2_branch_guard_snapshot_copies],
    pipe[:tree2_branch_guard_snapshot_ms],
    pipe[:tree2_branch_guard_snapshot_restore_ms],
    pipe[:tree2_branch_guard_snapshot_resync_base_ms],
    pipe[:tree2_branch_guard_suffix_replays],
    pipe[:tree2_branch_guard_suffix_replay_tokens],
    pipe[:tree2_branch_guard_suffix_replay_ms],
    pipe[:tree2_branch_guard_prefix_verify_ms],
    pipe[:tree2_branch_guard_prefix_verify_tokens],
    pipe[:tree2_branch_guard_token_verify_ms],
    pipe[:tree2_branch_guard_token_verify_tokens],
    pipe[:tree2_branch_guard_suffix_verify_ms],
    pipe[:tree2_branch_guard_suffix_verify_tokens],
    pipe[:tree2_branch_guard_snapshot_suffix_verify_ms],
    pipe[:tree2_branch_guard_snapshot_suffix_verify_tokens],
    pipe[:tree2_branch_guard_no_snapshot_suffix_verify_ms],
    pipe[:tree2_branch_guard_no_snapshot_suffix_verify_tokens],
    pipe[:risk_offramp_threshold],
    pipe[:risk_offramp_hits],
    pipe[:risk_offramp_delayed_blocks],
    pipe[:risk_offramp_delayed_tokens],
    pipe[:mtp_k2_reject_checks],
    pipe[:mtp_k2_reject_rescues],
    pipe[:mtp_k2_reject_misses],
    pipe[:mtp_k2_reject_ms],
    pipe[:reject_offramp_after],
    pipe[:reject_offramp_hits],
    pipe[:reject_offramp_tokens],
    pipe[:reject_offramp_ms])
end

private def build_self_spec_hybrid_routes(layer_ids : Array(Int32),
                                          manual_noffn : Set(Int32)?,
                                          manual_updown : Set(Int32)?,
                                          rich : Bool = false) : Array(HybridRoute)
  layers = layer_ids.uniq.sort
  routes = [] of HybridRoute
  seen = Set(String).new

  add_route = ->(name : String, noffn_values : Array(Int32), updown_values : Array(Int32)) {
    noffn_sorted = noffn_values.uniq.sort
    updown_sorted = updown_values.uniq.sort
    overlap = noffn_sorted & updown_sorted
    if overlap.empty?
      key = "#{noffn_sorted.join(',')}|#{updown_sorted.join(',')}"
      unless seen.includes?(key)
        seen << key
        noffn = noffn_sorted.empty? ? nil.as(Set(Int32)?) : Set(Int32).new(noffn_sorted).as(Set(Int32)?)
        updown = updown_sorted.empty? ? nil.as(Set(Int32)?) : Set(Int32).new(updown_sorted).as(Set(Int32)?)
        route_name = name.gsub(/[^A-Za-z0-9_.-]/, "_")
        routes << {name: route_name, noffn: noffn, updown: updown}
      end
    end
  }

  manual_noffn_values = manual_noffn ? manual_noffn.not_nil!.to_a : [] of Int32
  manual_updown_values = manual_updown ? manual_updown.not_nil!.to_a : [] of Int32
  # Emit the pure baseline before manual candidates so in-process scoreboards
  # compare routes against an earlier, not warmed-by-candidate, baseline.
  add_route.call("pure", [] of Int32, [] of Int32)
  add_route.call("manual", manual_noffn_values, manual_updown_values) unless manual_noffn_values.empty? && manual_updown_values.empty?
  return routes if layers.empty?

  head1 = layers[0, 1]
  head2 = layers[0, Math.min(2, layers.size)]
  tail1 = layers[layers.size - 1, 1]
  tail2_start = Math.max(layers.size - 2, 0)
  tail2 = layers[tail2_start, layers.size - tail2_start]

  add_route.call("noffn_#{head1.join('_')}", head1, [] of Int32)
  add_route.call("noffn_#{head2.join('_')}", head2, [] of Int32)
  add_route.call("updown_#{tail1.join('_')}", [] of Int32, tail1)
  add_route.call("updown_#{tail2.join('_')}", [] of Int32, tail2)
  add_route.call("hybrid_n#{head1.join('_')}_u#{tail1.join('_')}", head1, tail1)
  add_route.call("hybrid_n#{head1.join('_')}_u#{tail2.join('_')}", head1, tail2)
  add_route.call("hybrid_n#{head2.join('_')}_u#{tail2.join('_')}", head2, tail2)
  return routes unless rich

  layers.each do |il|
    add_route.call("noffn_single_#{il}", [il], [] of Int32)
    add_route.call("updown_single_#{il}", [] of Int32, [il])
  end

  max_group = Math.min(4, layers.size)
  (3..max_group).each do |n|
    prefix = layers[0, n]
    suffix = layers[layers.size - n, n]
    add_route.call("noffn_prefix#{n}", prefix, [] of Int32)
    add_route.call("updown_prefix#{n}", [] of Int32, prefix)
    add_route.call("noffn_suffix#{n}", suffix, [] of Int32)
    add_route.call("updown_suffix#{n}", [] of Int32, suffix)
  end

  (1..max_group).each do |n_count|
    (1..max_group).each do |u_count|
      noffn_prefix = layers[0, n_count]
      updown_suffix = layers[layers.size - u_count, u_count]
      add_route.call("hybrid_prefix#{n_count}_suffix#{u_count}", noffn_prefix, updown_suffix)
    end
  end

  even_slots = [] of Int32
  odd_slots = [] of Int32
  layers.each_with_index do |il, i|
    if i.even?
      even_slots << il
    else
      odd_slots << il
    end
  end
  add_route.call("hybrid_even_noffn_odd_updown", even_slots, odd_slots)
  add_route.call("hybrid_odd_noffn_even_updown", odd_slots, even_slots)
  routes
end

private def hybrid_route_note(route : HybridRoute, updown_rank : Int32?) : String
  noffn_note = route[:noffn] ? " draft_no_ffn_layers=#{route[:noffn].not_nil!.to_a.sort.join(',')}" : ""
  updown_note = (updown_rank && route[:updown]) ? " draft_pca_updown_layers=#{route[:updown].not_nil!.to_a.sort.join(',')}" : ""
  " hybrid_route=#{route[:name]}#{noffn_note}#{updown_note}"
end

private def append_route_score(rows : Array(RouteScoreRow),
                               prompt_name : String,
                               mode : String,
                               route : HybridRoute,
                               draft_split : Int32?,
                               updown_rank : Int32?,
                               pipe,
                               accept_rate : Float64,
                               residual_mean : Float64? = nil,
                               residual_p90 : Float64? = nil,
                               residual_max : Float64? = nil,
                               repeat_rate : Float64? = nil,
                               bigram_repeat_rate : Float64? = nil,
                               unique_rate : Float64? = nil)
  rows << {
    prompt:                  prompt_name,
    mode:                    mode,
    split:                   draft_split.nil? ? "nil" : draft_split.to_s,
    route:                   route[:name],
    updown_rank:             updown_rank,
    parity:                  pipe[:parity],
    accept_rate:             accept_rate,
    rejections:              pipe[:rejections],
    plain_speedup:           pipe[:plain_speedup],
    overlap_ms:              pipe[:overlap_ms],
    plain_exact_ms:          pipe[:plain_exact_ms],
    draft_wait_ms:           pipe[:draft_wait_ms],
    replay_ms:               pipe[:replay_ms],
    tree2_margin_min:        pipe[:tree2_margin_min],
    tree2_reject_margin_min: pipe[:tree2_reject_margin_min],
    residual_mean:           residual_mean,
    residual_p90:            residual_p90,
    residual_max:            residual_max,
    repeat_rate:             repeat_rate,
    bigram_repeat_rate:      bigram_repeat_rate,
    unique_rate:             unique_rate,
  }
end

private def route_baseline_key(row : RouteScoreRow) : String
  "#{row[:prompt]}|#{row[:mode]}|#{row[:split]}"
end

private def route_stability_key(row : RouteScoreRow) : String
  updown = row[:updown_rank] ? row[:updown_rank].to_s : "-"
  "#{row[:mode]}|#{row[:split]}|#{row[:route]}|#{updown}"
end

private def risk_offramp_threshold_label(value : Float64?) : String
  value.nil? ? "baseline" : value.not_nil!.to_s
end

private def median_float(values : Array(Float64)) : Float64
  return 0.0 if values.empty?
  sorted = values.sort
  mid = sorted.size // 2
  sorted.size.odd? ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) * 0.5
end

private def draft_body_label(rank : Int32?) : String
  rank ? "pca_updown#{rank}" : "lowrank"
end

private def append_draft_body_score(rows : Array(DraftBodyScoreRow),
                                    prompt_name : String,
                                    mode : String,
                                    draft_split : Int32?,
                                    updown_rank : Int32?,
                                    pipe,
                                    accept_rate : Float64)
  rows << {
    prompt:               prompt_name,
    mode:                 mode,
    split:                draft_split.nil? ? "nil" : draft_split.to_s,
    body:                 draft_body_label(updown_rank),
    updown_rank:          updown_rank,
    parity:               pipe[:parity],
    accept_rate:          accept_rate,
    rejections:           pipe[:rejections],
    draft_updown_chunks:  pipe[:draft_updown_chunks],
    plain_speedup:        pipe[:plain_speedup],
    overlap_ms:           pipe[:overlap_ms],
    plain_exact_ms:       pipe[:plain_exact_ms],
    draft_next_ms:        pipe[:draft_next_ms],
    verifier_ms:          pipe[:verifier_ms],
    draft_wait_ms:        pipe[:draft_wait_ms],
    replay_ms:            pipe[:replay_ms],
  }
end

private def draft_body_baseline_key(row : DraftBodyScoreRow) : String
  "#{row[:prompt]}|#{row[:mode]}|#{row[:split]}"
end

private def draft_body_group_key(row : DraftBodyScoreRow) : String
  updown = row[:updown_rank] ? row[:updown_rank].to_s : "-"
  "#{row[:mode]}|#{row[:split]}|#{row[:body]}|#{updown}"
end

private def draft_body_baseline_groups(rows : Array(DraftBodyScoreRow))
  baselines = Hash(String, Array(DraftBodyScoreRow)).new { |h, k| h[k] = [] of DraftBodyScoreRow }
  rows.each do |row|
    next unless row[:parity]
    next unless row[:updown_rank].nil?
    baselines[draft_body_baseline_key(row)] << row
  end
  baselines
end

private def draft_body_baseline_sample_count(groups : Hash(String, Array(DraftBodyScoreRow))) : Int32
  total = 0
  groups.each_value { |group| total += group.size }
  total
end

private def draft_body_baseline_overlap_ms(groups : Hash(String, Array(DraftBodyScoreRow)), key : String) : Float64?
  group = groups[key]?
  return nil unless group && !group.empty?
  median_float(group.map { |row| row[:overlap_ms] })
end

private def draft_body_baseline_replay_ms(groups : Hash(String, Array(DraftBodyScoreRow)), key : String) : Float64?
  group = groups[key]?
  return nil unless group && !group.empty?
  median_float(group.map { |row| row[:replay_ms] })
end

private def draft_body_baseline_rejections(groups : Hash(String, Array(DraftBodyScoreRow)), key : String) : Float64?
  group = groups[key]?
  return nil unless group && !group.empty?
  median_float(group.map { |row| row[:rejections].to_f64 })
end

private def draft_body_baseline_accept_rate(groups : Hash(String, Array(DraftBodyScoreRow)), key : String) : Float64?
  group = groups[key]?
  return nil unless group && !group.empty?
  median_float(group.map { |row| row[:accept_rate] })
end

private def print_draft_body_scoreboard(rows : Array(DraftBodyScoreRow), limit : Int32 = 40)
  return if rows.empty?
  baselines = draft_body_baseline_groups(rows)
  baseline_samples = draft_body_baseline_sample_count(baselines)
  ranked = rows.sort do |a, b|
    a_base = draft_body_baseline_overlap_ms(baselines, draft_body_baseline_key(a))
    b_base = draft_body_baseline_overlap_ms(baselines, draft_body_baseline_key(b))
    a_delta = a_base && a_base > 0.0 ? ((a_base - a[:overlap_ms]) * 100.0 / a_base) : 0.0
    b_delta = b_base && b_base > 0.0 ? ((b_base - b[:overlap_ms]) * 100.0 / b_base) : 0.0
    b_delta <=> a_delta
  end

  puts "self_spec_draft_body_scoreboard rows=#{rows.size} baseline_groups=#{baselines.size} baseline_samples=#{baseline_samples} limit=#{limit}"
  puts "rank prompt mode split body updown parity accept% accept_delta rejections reject_delta draft_updown_chunks plain_speedup overlap_ms baseline_delta% draft_next_ms verifier_ms draft_wait_ms replay_ms replay_delta_ms"
  ranked.first(limit).each_with_index do |row, i|
    baseline_key = draft_body_baseline_key(row)
    baseline_overlap = draft_body_baseline_overlap_ms(baselines, baseline_key)
    baseline_replay = draft_body_baseline_replay_ms(baselines, baseline_key)
    baseline_rejections = draft_body_baseline_rejections(baselines, baseline_key)
    baseline_accept = draft_body_baseline_accept_rate(baselines, baseline_key)
    baseline_delta = baseline_overlap && baseline_overlap > 0.0 ? ((baseline_overlap - row[:overlap_ms]) * 100.0 / baseline_overlap) : nil
    replay_delta = baseline_replay ? row[:replay_ms] - baseline_replay : nil
    reject_delta = baseline_rejections ? row[:rejections].to_f64 - baseline_rejections : nil
    accept_delta = baseline_accept ? row[:accept_rate] - baseline_accept : nil
    updown_text = row[:updown_rank] ? row[:updown_rank].to_s : "-"
    puts "#{i + 1} #{row[:prompt]} #{row[:mode]} #{row[:split]} #{row[:body]} #{updown_text} #{row[:parity]} #{row[:accept_rate].round(2)} #{accept_delta ? sprintf("%.2f", accept_delta) : "na"} #{row[:rejections]} #{reject_delta ? sprintf("%.2f", reject_delta) : "na"} #{row[:draft_updown_chunks]} #{row[:plain_speedup].round(4)} #{row[:overlap_ms].round(3)} #{baseline_delta ? sprintf("%.2f", baseline_delta) : "na"} #{row[:draft_next_ms].round(3)} #{row[:verifier_ms].round(3)} #{row[:draft_wait_ms].round(3)} #{row[:replay_ms].round(3)} #{replay_delta ? sprintf("%.3f", replay_delta) : "na"}"
  end
end

private def print_draft_body_stability_scoreboard(rows : Array(DraftBodyScoreRow), limit : Int32 = 20)
  return if rows.empty?
  baselines = draft_body_baseline_groups(rows)
  baseline_samples = draft_body_baseline_sample_count(baselines)
  groups = Hash(String, Array(DraftBodyScoreRow)).new { |h, k| h[k] = [] of DraftBodyScoreRow }
  rows.each { |row| groups[draft_body_group_key(row)] << row }

  summaries = [] of NamedTuple(key: String, rows: Int32, baseline_samples: Int32, parity_all: Bool, accept_mean: Float64, accept_delta_mean: Float64, plain_speedup_mean: Float64, overlap_total: Float64, delta_mean: Float64, delta_min: Float64, replay_delta_mean: Float64, reject_delta_total: Float64, draft_updown_chunks: Int32, score: Float64)
  groups.each do |key, group|
    row_count = group.size
    parity_all = group.all? { |row| row[:parity] }
    accept_mean = group.sum { |row| row[:accept_rate] } / row_count
    plain_speedup_mean = group.sum { |row| row[:plain_speedup] } / row_count
    overlap_total = group.sum { |row| row[:overlap_ms] }
    draft_updown_chunks = group.sum { |row| row[:draft_updown_chunks] }
    deltas = [] of Float64
    accept_deltas = [] of Float64
    replay_deltas = [] of Float64
    reject_delta_total = 0.0
    baseline_count = 0
    seen_baseline_keys = Set(String).new
    group.each do |row|
      baseline_key = draft_body_baseline_key(row)
      if baseline_group = baselines[baseline_key]?
        if seen_baseline_keys.add?(baseline_key)
          baseline_count += baseline_group.size
        end
        baseline_overlap = draft_body_baseline_overlap_ms(baselines, baseline_key)
        baseline_replay = draft_body_baseline_replay_ms(baselines, baseline_key)
        baseline_rejections = draft_body_baseline_rejections(baselines, baseline_key)
        baseline_accept = draft_body_baseline_accept_rate(baselines, baseline_key)
        deltas << ((baseline_overlap - row[:overlap_ms]) * 100.0 / baseline_overlap) if baseline_overlap && baseline_overlap > 0.0
        replay_deltas << (row[:replay_ms] - baseline_replay) if baseline_replay
        reject_delta_total += row[:rejections].to_f64 - baseline_rejections if baseline_rejections
        accept_deltas << (row[:accept_rate] - baseline_accept) if baseline_accept
      end
    end
    delta_mean = deltas.empty? ? 0.0 : deltas.sum / deltas.size
    delta_min = deltas.empty? ? 0.0 : deltas.min
    replay_delta_mean = replay_deltas.empty? ? 0.0 : replay_deltas.sum / replay_deltas.size
    accept_delta_mean = accept_deltas.empty? ? 0.0 : accept_deltas.sum / accept_deltas.size
    score = parity_all ? (delta_mean + delta_min * 0.25 + accept_delta_mean * 0.1 - [replay_delta_mean, 0.0].max / 1000.0 - [reject_delta_total, 0.0].max * 0.5) : -1.0e9
    summaries << {
      key:                 key,
      rows:                row_count,
      baseline_samples:    baseline_count,
      parity_all:          parity_all,
      accept_mean:         accept_mean,
      accept_delta_mean:   accept_delta_mean,
      plain_speedup_mean:  plain_speedup_mean,
      overlap_total:       overlap_total,
      delta_mean:          delta_mean,
      delta_min:           delta_min,
      replay_delta_mean:   replay_delta_mean,
      reject_delta_total:  reject_delta_total,
      draft_updown_chunks: draft_updown_chunks,
      score:               score,
    }
  end

  ranked = summaries.sort { |a, b| b[:score] <=> a[:score] }
  puts "self_spec_draft_body_stability_scoreboard groups=#{summaries.size} baseline_groups=#{baselines.size} baseline_samples=#{baseline_samples} limit=#{limit}"
  puts "rank mode split body updown rows baseline_samples parity_all accept_mean accept_delta_mean plain_speedup_mean overlap_total baseline_delta_mean% baseline_delta_min% replay_delta_mean_ms reject_delta_total draft_updown_chunks score"
  ranked.first(limit).each_with_index do |row, i|
    mode, split, body, updown = row[:key].split('|')
    puts "#{i + 1} #{mode} #{split} #{body} #{updown} #{row[:rows]} #{row[:baseline_samples]} #{row[:parity_all]} #{row[:accept_mean].round(2)} #{row[:accept_delta_mean].round(2)} #{row[:plain_speedup_mean].round(4)} #{row[:overlap_total].round(3)} #{row[:delta_mean].round(2)} #{row[:delta_min].round(2)} #{row[:replay_delta_mean].round(3)} #{row[:reject_delta_total].round(2)} #{row[:draft_updown_chunks]} #{row[:score].round(4)}"
  end
end

private def append_risk_offramp_score(rows : Array(RiskOfframpScoreRow),
                                      prompt_name : String,
                                      mode : String,
                                      draft_split : Int32?,
                                      threshold : Float64?,
                                      pipe,
                                      accept_rate : Float64)
  rows << {
    prompt:            prompt_name,
    mode:              mode,
    split:             draft_split.nil? ? "nil" : draft_split.to_s,
    threshold:         risk_offramp_threshold_label(threshold),
    parity:            pipe[:parity],
    accept_rate:       accept_rate,
    rejections:        pipe[:rejections],
    plain_speedup:     pipe[:plain_speedup],
    overlap_ms:        pipe[:overlap_ms],
    plain_exact_ms:    pipe[:plain_exact_ms],
    draft_wait_ms:     pipe[:draft_wait_ms],
    replay_ms:         pipe[:replay_ms],
    risk_hits:         pipe[:risk_offramp_hits],
    delayed_blocks:    pipe[:risk_offramp_delayed_blocks],
    delayed_tokens:    pipe[:risk_offramp_delayed_tokens],
    margin_min:        pipe[:tree2_margin_min],
    reject_margin_min: pipe[:tree2_reject_margin_min],
  }
end

private def risk_offramp_baseline_key(row : RiskOfframpScoreRow) : String
  "#{row[:prompt]}|#{row[:mode]}|#{row[:split]}"
end

private def risk_offramp_group_key(row : RiskOfframpScoreRow) : String
  "#{row[:mode]}|#{row[:split]}|#{row[:threshold]}"
end

private def risk_offramp_baseline_groups(rows : Array(RiskOfframpScoreRow))
  baselines = Hash(String, Array(RiskOfframpScoreRow)).new { |h, k| h[k] = [] of RiskOfframpScoreRow }
  rows.each do |row|
    next unless row[:parity]
    next unless row[:threshold] == "baseline"
    baselines[risk_offramp_baseline_key(row)] << row
  end
  baselines
end

private def risk_offramp_baseline_sample_count(groups : Hash(String, Array(RiskOfframpScoreRow))) : Int32
  total = 0
  groups.each_value { |group| total += group.size }
  total
end

private def risk_offramp_baseline_overlap_ms(groups : Hash(String, Array(RiskOfframpScoreRow)), key : String) : Float64?
  group = groups[key]?
  return nil unless group && !group.empty?
  median_float(group.map { |row| row[:overlap_ms] })
end

private def risk_offramp_baseline_replay_ms(groups : Hash(String, Array(RiskOfframpScoreRow)), key : String) : Float64?
  group = groups[key]?
  return nil unless group && !group.empty?
  median_float(group.map { |row| row[:replay_ms] })
end

private def risk_offramp_baseline_rejections(groups : Hash(String, Array(RiskOfframpScoreRow)), key : String) : Float64?
  group = groups[key]?
  return nil unless group && !group.empty?
  median_float(group.map { |row| row[:rejections].to_f64 })
end

private def print_risk_offramp_scoreboard(rows : Array(RiskOfframpScoreRow), limit : Int32 = 40)
  return if rows.empty?
  baselines = risk_offramp_baseline_groups(rows)
  baseline_samples = risk_offramp_baseline_sample_count(baselines)

  ranked = rows.sort do |a, b|
    a_base = risk_offramp_baseline_overlap_ms(baselines, risk_offramp_baseline_key(a))
    b_base = risk_offramp_baseline_overlap_ms(baselines, risk_offramp_baseline_key(b))
    a_delta = a_base && a_base > 0.0 ? ((a_base - a[:overlap_ms]) * 100.0 / a_base) : 0.0
    b_delta = b_base && b_base > 0.0 ? ((b_base - b[:overlap_ms]) * 100.0 / b_base) : 0.0
    b_delta <=> a_delta
  end

  puts "self_spec_risk_offramp_scoreboard rows=#{rows.size} baseline_groups=#{baselines.size} baseline_samples=#{baseline_samples} limit=#{limit}"
  puts "rank prompt mode split threshold parity accept% plain_speedup overlap_ms baseline_delta% draft_wait_ms replay_ms replay_delta_ms rejections reject_delta risk_hits delayed_blocks delayed_tokens margin_min reject_margin_min"
  ranked.first(limit).each_with_index do |row, i|
    baseline_key = risk_offramp_baseline_key(row)
    baseline_overlap = risk_offramp_baseline_overlap_ms(baselines, baseline_key)
    baseline_replay = risk_offramp_baseline_replay_ms(baselines, baseline_key)
    baseline_rejections = risk_offramp_baseline_rejections(baselines, baseline_key)
    baseline_delta = baseline_overlap && baseline_overlap > 0.0 ? ((baseline_overlap - row[:overlap_ms]) * 100.0 / baseline_overlap) : nil
    replay_delta = baseline_replay ? row[:replay_ms] - baseline_replay : nil
    reject_delta = baseline_rejections ? row[:rejections].to_f64 - baseline_rejections : nil
    delta_text = baseline_delta ? sprintf("%.2f", baseline_delta) : "na"
    replay_delta_text = replay_delta ? sprintf("%.3f", replay_delta) : "na"
    reject_delta_text = reject_delta ? sprintf("%.2f", reject_delta) : "na"
    puts "#{i + 1} #{row[:prompt]} #{row[:mode]} #{row[:split]} #{row[:threshold]} #{row[:parity]} #{row[:accept_rate].round(2)} #{row[:plain_speedup].round(4)} #{row[:overlap_ms].round(3)} #{delta_text} #{row[:draft_wait_ms].round(3)} #{row[:replay_ms].round(3)} #{replay_delta_text} #{row[:rejections]} #{reject_delta_text} #{row[:risk_hits]} #{row[:delayed_blocks]} #{row[:delayed_tokens]} #{row[:margin_min].round(4)} #{row[:reject_margin_min].round(4)}"
  end
end

private def print_risk_offramp_stability_scoreboard(rows : Array(RiskOfframpScoreRow), limit : Int32 = 20)
  return if rows.empty?
  baselines = risk_offramp_baseline_groups(rows)
  baseline_samples = risk_offramp_baseline_sample_count(baselines)

  groups = Hash(String, Array(RiskOfframpScoreRow)).new { |h, k| h[k] = [] of RiskOfframpScoreRow }
  rows.each { |row| groups[risk_offramp_group_key(row)] << row }

  summaries = [] of NamedTuple(key: String, rows: Int32, baseline_samples: Int32, parity_all: Bool, accept_mean: Float64, delta_mean: Float64, delta_min: Float64, delta_max: Float64, plain_speedup_mean: Float64, replay_delta_mean: Float64, reject_delta_total: Float64, false_offramp_hits: Int32, risk_hits: Int32, delayed_tokens: Int32, overlap_total: Float64, score: Float64)
  groups.each do |key, group|
    row_count = group.size
    parity_all = group.all? { |row| row[:parity] }
    accept_mean = group.sum { |row| row[:accept_rate] } / row_count
    plain_speedup_mean = group.sum { |row| row[:plain_speedup] } / row_count
    overlap_total = group.sum { |row| row[:overlap_ms] }
    deltas = [] of Float64
    replay_deltas = [] of Float64
    reject_delta_total = 0.0
    false_offramp_hits = 0
    baseline_count = 0
    seen_baseline_keys = Set(String).new
    group.each do |row|
      baseline_key = risk_offramp_baseline_key(row)
      if baseline_group = baselines[baseline_key]?
        if seen_baseline_keys.add?(baseline_key)
          baseline_count += baseline_group.size
        end
        baseline_overlap = risk_offramp_baseline_overlap_ms(baselines, baseline_key)
        baseline_replay = risk_offramp_baseline_replay_ms(baselines, baseline_key)
        baseline_rejections = risk_offramp_baseline_rejections(baselines, baseline_key)
        deltas << ((baseline_overlap - row[:overlap_ms]) * 100.0 / baseline_overlap) if baseline_overlap && baseline_overlap > 0.0
        replay_deltas << (row[:replay_ms] - baseline_replay) if baseline_replay
        reject_delta_total += row[:rejections].to_f64 - baseline_rejections if baseline_rejections
        false_offramp_hits += row[:risk_hits] if baseline_rejections && baseline_rejections <= 0.0 && row[:rejections] == 0
      end
    end
    delta_mean = deltas.empty? ? 0.0 : deltas.sum / deltas.size
    delta_min = deltas.empty? ? 0.0 : deltas.min
    delta_max = deltas.empty? ? 0.0 : deltas.max
    replay_delta_mean = replay_deltas.empty? ? 0.0 : replay_deltas.sum / replay_deltas.size
    risk_hits = group.sum { |row| row[:risk_hits] }
    delayed_tokens = group.sum { |row| row[:delayed_tokens] }
    score = parity_all ? (delta_mean + delta_min * 0.25 - false_offramp_hits * 0.25 - [replay_delta_mean, 0.0].max / 1000.0) : -1.0e9
    summaries << {
      key:                key,
      rows:               row_count,
      baseline_samples:   baseline_count,
      parity_all:         parity_all,
      accept_mean:        accept_mean,
      delta_mean:         delta_mean,
      delta_min:          delta_min,
      delta_max:          delta_max,
      plain_speedup_mean: plain_speedup_mean,
      replay_delta_mean:  replay_delta_mean,
      reject_delta_total: reject_delta_total,
      false_offramp_hits: false_offramp_hits,
      risk_hits:          risk_hits,
      delayed_tokens:     delayed_tokens,
      overlap_total:      overlap_total,
      score:              score,
    }
  end

  ranked = summaries.sort { |a, b| b[:score] <=> a[:score] }
  puts "self_spec_risk_offramp_stability_scoreboard groups=#{summaries.size} baseline_groups=#{baselines.size} baseline_samples=#{baseline_samples} limit=#{limit}"
  puts "rank mode split threshold rows baseline_samples parity_all accept_mean plain_speedup_mean overlap_total baseline_delta_mean% baseline_delta_min% baseline_delta_max% replay_delta_mean_ms reject_delta_total false_offramp_hits risk_hits delayed_tokens score"
  ranked.first(limit).each_with_index do |row, i|
    mode, split, threshold = row[:key].split('|')
    puts "#{i + 1} #{mode} #{split} #{threshold} #{row[:rows]} #{row[:baseline_samples]} #{row[:parity_all]} #{row[:accept_mean].round(2)} #{row[:plain_speedup_mean].round(4)} #{row[:overlap_total].round(3)} #{row[:delta_mean].round(2)} #{row[:delta_min].round(2)} #{row[:delta_max].round(2)} #{row[:replay_delta_mean].round(3)} #{row[:reject_delta_total].round(2)} #{row[:false_offramp_hits]} #{row[:risk_hits]} #{row[:delayed_tokens]} #{row[:score].round(4)}"
  end
end

private def route_score(row : RouteScoreRow, baseline_overlap : Float64?) : Float64
  return -1.0e9 unless row[:parity]
  speed_component = baseline_overlap && baseline_overlap > 0.0 ? baseline_overlap / row[:overlap_ms] : row[:plain_speedup]
  speed_component + (row[:accept_rate] / 1000.0) - (row[:replay_ms] / 10000.0)
end

private def optional_route_float(value : Float64?) : String
  value.nil? ? "na" : value.not_nil!.round(4).to_s
end

private def print_route_scoreboard(rows : Array(RouteScoreRow), limit : Int32 = 30)
  return if rows.empty?
  baselines = {} of String => Float64
  rows.each do |row|
    next unless row[:parity]
    next unless row[:route] == "pure" && row[:updown_rank].nil?
    baselines[route_baseline_key(row)] = row[:overlap_ms]
  end
  ranked = rows.sort do |a, b|
    route_score(b, baselines[route_baseline_key(b)]?) <=> route_score(a, baselines[route_baseline_key(a)]?)
  end
  puts "self_spec_route_scoreboard rows=#{rows.size} baselines=#{baselines.size} limit=#{limit}"
  puts "rank prompt mode split route updown parity accept% plain_speedup overlap_ms baseline_delta% draft_wait_ms replay_ms margin_min reject_margin_min rejections"
  ranked.first(limit).each_with_index do |row, i|
    baseline = baselines[route_baseline_key(row)]?
    baseline_delta = baseline && baseline > 0.0 ? ((baseline - row[:overlap_ms]) * 100.0 / baseline) : nil
    delta_text = baseline_delta ? sprintf("%.2f", baseline_delta) : "na"
    updown_text = row[:updown_rank] ? row[:updown_rank].to_s : "-"
    puts "#{i + 1} #{row[:prompt]} #{row[:mode]} #{row[:split]} #{row[:route]} #{updown_text} #{row[:parity]} #{row[:accept_rate].round(2)} #{row[:plain_speedup].round(4)} #{row[:overlap_ms].round(3)} #{delta_text} #{row[:draft_wait_ms].round(3)} #{row[:replay_ms].round(3)} #{row[:tree2_margin_min].round(4)} #{row[:tree2_reject_margin_min].round(4)} #{row[:rejections]}"
  end
end

private def print_route_stability_scoreboard(rows : Array(RouteScoreRow), limit : Int32 = 30)
  return if rows.empty?
  baselines = {} of String => Float64
  rows.each do |row|
    next unless row[:parity]
    next unless row[:route] == "pure" && row[:updown_rank].nil?
    baselines[route_baseline_key(row)] = row[:overlap_ms]
  end

  groups = Hash(String, Array(RouteScoreRow)).new { |h, k| h[k] = [] of RouteScoreRow }
  rows.each do |row|
    groups[route_stability_key(row)] << row
  end

  summaries = [] of NamedTuple(
    key: String,
    prompts: Int32,
    baseline_count: Int32,
    parity_all: Bool,
    accept_mean: Float64,
    plain_speedup_mean: Float64,
    overlap_total: Float64,
    delta_mean: Float64,
    delta_min: Float64,
    delta_max: Float64,
    wins: Int32,
    losses: Int32,
    ties: Int32,
    max_loss: Float64,
    replay_max: Float64,
    margin_min: Float64,
    score: Float64)
  groups.each do |key, group|
    prompts = group.size
    parity_all = group.all? { |row| row[:parity] }
    accept_mean = group.sum { |row| row[:accept_rate] } / prompts
    plain_speedup_mean = group.sum { |row| row[:plain_speedup] } / prompts
    overlap_total = group.sum { |row| row[:overlap_ms] }
    replay_max = group.max_of { |row| row[:replay_ms] }
    margin_min = group.min_of { |row| row[:tree2_margin_min] }
    deltas = [] of Float64
    group.each do |row|
      if baseline = baselines[route_baseline_key(row)]?
        deltas << ((baseline - row[:overlap_ms]) * 100.0 / baseline) if baseline > 0.0
      end
    end
    baseline_count = deltas.size
    delta_mean = deltas.empty? ? 0.0 : deltas.sum / deltas.size
    delta_min = deltas.empty? ? 0.0 : deltas.min
    delta_max = deltas.empty? ? 0.0 : deltas.max
    wins = deltas.count { |delta| delta > 0.5 }
    losses = deltas.count { |delta| delta < -0.5 }
    ties = deltas.size - wins - losses
    max_loss = deltas.empty? ? 0.0 : [0.0, -delta_min].max
    speed_score = baseline_count > 0 ? delta_mean : (plain_speedup_mean * 100.0)
    # Penalize unsafe tails explicitly: route selection needs stable wins, not just a good mean.
    score = parity_all ? (speed_score + delta_min * 0.5 + accept_mean / 100.0 - replay_max / 1000.0 - losses * 10.0 - max_loss * 0.25) : -1.0e9
    summaries << {
      key:                key,
      prompts:            prompts,
      baseline_count:     baseline_count,
      parity_all:         parity_all,
      accept_mean:        accept_mean,
      plain_speedup_mean: plain_speedup_mean,
      overlap_total:      overlap_total,
      delta_mean:         delta_mean,
      delta_min:          delta_min,
      delta_max:          delta_max,
      wins:               wins,
      losses:             losses,
      ties:               ties,
      max_loss:           max_loss,
      replay_max:         replay_max,
      margin_min:         margin_min,
      score:              score,
    }
  end

  ranked = summaries.sort { |a, b| b[:score] <=> a[:score] }
  puts "self_spec_route_stability_scoreboard groups=#{summaries.size} baselines=#{baselines.size} limit=#{limit}"
  puts "rank mode split route updown prompts baselines parity_all accept_mean plain_speedup_mean overlap_total baseline_delta_mean% baseline_delta_min% baseline_delta_max% wins losses ties max_loss% replay_max margin_min score"
  ranked.first(limit).each_with_index do |row, i|
    mode, split, route, updown = row[:key].split('|')
    puts "#{i + 1} #{mode} #{split} #{route} #{updown} #{row[:prompts]} #{row[:baseline_count]} #{row[:parity_all]} #{row[:accept_mean].round(2)} #{row[:plain_speedup_mean].round(4)} #{row[:overlap_total].round(3)} #{row[:delta_mean].round(2)} #{row[:delta_min].round(2)} #{row[:delta_max].round(2)} #{row[:wins]} #{row[:losses]} #{row[:ties]} #{row[:max_loss].round(2)} #{row[:replay_max].round(3)} #{row[:margin_min].round(4)} #{row[:score].round(4)}"
  end
end

private def print_route_oracle_scoreboard(rows : Array(RouteScoreRow), limit : Int32 = 30)
  return if rows.empty?
  baselines = {} of String => RouteScoreRow
  rows.each do |row|
    next unless row[:parity]
    next unless row[:route] == "pure" && row[:updown_rank].nil?
    baselines[route_baseline_key(row)] = row
  end
  return if baselines.empty?

  groups = Hash(String, Array(RouteScoreRow)).new { |h, k| h[k] = [] of RouteScoreRow }
  rows.each do |row|
    next unless row[:parity]
    key = route_baseline_key(row)
    next unless baselines.has_key?(key)
    groups[key] << row
  end

  picks = [] of NamedTuple(prompt: String, mode: String, split: String, route: String, updown: String, accept_rate: Float64, baseline_ms: Float64, best_ms: Float64, delta: Float64, replay_ms: Float64, margin_min: Float64, reject_margin_min: Float64, rejections: Int32, residual_mean: Float64?, residual_p90: Float64?, residual_max: Float64?, repeat_rate: Float64?, bigram_repeat_rate: Float64?, unique_rate: Float64?)
  pure_total = 0.0
  best_total = 0.0
  groups.each do |key, group|
    baseline = baselines[key]
    best = group.min_by { |row| row[:overlap_ms] }
    next unless baseline[:overlap_ms] > 0.0
    delta = (baseline[:overlap_ms] - best[:overlap_ms]) * 100.0 / baseline[:overlap_ms]
    pure_total += baseline[:overlap_ms]
    best_total += best[:overlap_ms]
    picks << {
      prompt:            best[:prompt],
      mode:              best[:mode],
      split:             best[:split],
      route:             best[:route],
      updown:            best[:updown_rank] ? best[:updown_rank].to_s : "-",
      accept_rate:       best[:accept_rate],
      baseline_ms:       baseline[:overlap_ms],
      best_ms:           best[:overlap_ms],
      delta:             delta,
      replay_ms:         best[:replay_ms],
      margin_min:        best[:tree2_margin_min],
      reject_margin_min: best[:tree2_reject_margin_min],
      rejections:        best[:rejections],
      residual_mean:     best[:residual_mean],
      residual_p90:      best[:residual_p90],
      residual_max:      best[:residual_max],
      repeat_rate:       best[:repeat_rate],
      bigram_repeat_rate: best[:bigram_repeat_rate],
      unique_rate:       best[:unique_rate],
    }
  end

  total_delta = pure_total > 0.0 ? (pure_total - best_total) * 100.0 / pure_total : 0.0
  puts "self_spec_route_oracle prompts=#{picks.size} pure_overlap_total=#{pure_total.round(3)} oracle_overlap_total=#{best_total.round(3)} oracle_delta%=#{total_delta.round(2)} limit=#{limit}"
  puts "rank prompt mode split best_route updown accept% baseline_ms best_ms delta% replay_ms margin_min reject_margin_min rejections residual_mean residual_p90 residual_max repeat_rate bigram_repeat_rate unique_rate"
  picks.sort_by { |row| -row[:delta] }.first(limit).each_with_index do |row, i|
    puts "#{i + 1} #{row[:prompt]} #{row[:mode]} #{row[:split]} #{row[:route]} #{row[:updown]} #{row[:accept_rate].round(2)} #{row[:baseline_ms].round(3)} #{row[:best_ms].round(3)} #{row[:delta].round(2)} #{row[:replay_ms].round(3)} #{row[:margin_min].round(4)} #{row[:reject_margin_min].round(4)} #{row[:rejections]} #{optional_route_float(row[:residual_mean])} #{optional_route_float(row[:residual_p90])} #{optional_route_float(row[:residual_max])} #{optional_route_float(row[:repeat_rate])} #{optional_route_float(row[:bigram_repeat_rate])} #{optional_route_float(row[:unique_rate])}"
  end
end

private def route_feature_value(row : RouteScoreRow, feature : String) : Float64?
  case feature
  when "residual_mean"
    row[:residual_mean]
  when "residual_p90"
    row[:residual_p90]
  when "residual_max"
    row[:residual_max]
  when "repeat_rate"
    row[:repeat_rate]
  when "bigram_repeat_rate"
    row[:bigram_repeat_rate]
  when "unique_rate"
    row[:unique_rate]
  else
    nil
  end
end

private def prompt_route_selector_feature_value(residual_stats, value_stats, feature : String) : Float64?
  case feature
  when "residual_mean"
    residual_stats[:mean]
  when "residual_p90"
    residual_stats[:p90]
  when "residual_max"
    residual_stats[:max]
  when "repeat_rate"
    value_stats[:repeat_rate]
  when "bigram_repeat_rate"
    value_stats[:bigram_repeat_rate]
  when "unique_rate"
    value_stats[:unique_rate]
  else
    nil
  end
end

private def route_selector_match?(value : Float64?, op : String, threshold : Float64?) : Bool
  return false unless value && threshold
  case op
  when "<="
    value <= threshold
  when ">="
    value >= threshold
  else
    false
  end
end

private def route_selector_note(route_name : String,
                                feature : String,
                                op : String,
                                threshold : Float64?,
                                value : Float64?,
                                selected : Bool,
                                selected_route : HybridRoute) : String
  threshold_label = threshold.nil? ? "none" : threshold.not_nil!.round(6).to_s
  value_label = value.nil? ? "none" : value.not_nil!.round(6).to_s
  " route_selector=#{selected ? "select" : "pure"} route_selector_route=#{route_name} route_selector_selected_route=#{selected_route[:name]} route_selector_feature=#{feature} route_selector_op=#{op} route_selector_threshold=#{threshold_label} route_selector_value=#{value_label}"
end

private def print_route_selector_scoreboard(rows : Array(RouteScoreRow), limit : Int32 = 30)
  return if rows.empty?
  baselines = {} of String => RouteScoreRow
  rows.each do |row|
    next unless row[:parity]
    next unless row[:route] == "pure" && row[:updown_rank].nil?
    baselines[route_baseline_key(row)] = row
  end
  return if baselines.empty?

  groups = Hash(String, Array(RouteScoreRow)).new { |h, k| h[k] = [] of RouteScoreRow }
  rows.each do |row|
    next unless row[:parity]
    next if row[:route] == "pure" && row[:updown_rank].nil?
    next unless baselines.has_key?(route_baseline_key(row))
    groups[route_stability_key(row)] << row
  end
  return if groups.empty?

  features = ["residual_mean", "residual_p90", "residual_max", "repeat_rate", "bigram_repeat_rate", "unique_rate"]
  policies = [] of NamedTuple(key: String, feature: String, op: String, threshold: Float64, prompts: Int32, selected: Int32, wins: Int32, losses: Int32, ties: Int32, baseline_total: Float64, policy_total: Float64, delta: Float64, worst_delta: Float64, max_loss: Float64, score: Float64)
  groups.each do |key, group|
    row_by_baseline = {} of String => RouteScoreRow
    group.each { |row| row_by_baseline[route_baseline_key(row)] = row }
    mode, split, _route, _updown = key.split('|')
    candidate_baselines = baselines.select { |_base_key, base| base[:mode] == mode && base[:split] == split }
    next if candidate_baselines.empty?

    features.each do |feature|
      thresholds = group.compact_map { |row| route_feature_value(row, feature) }.uniq.sort
      thresholds.each do |threshold|
        {"<=", ">="}.each do |op|
          baseline_total = 0.0
          policy_total = 0.0
          selected = 0
          wins = 0
          losses = 0
          ties = 0
          selected_deltas = [] of Float64
          candidate_baselines.each do |base_key, baseline|
            baseline_total += baseline[:overlap_ms]
            candidate = row_by_baseline[base_key]?
            feature_value = candidate ? route_feature_value(candidate, feature) : nil
            use_candidate = false
            if candidate && feature_value
              use_candidate = op == "<=" ? feature_value <= threshold : feature_value >= threshold
            end
            if use_candidate && candidate
              selected += 1
              policy_total += candidate[:overlap_ms]
              if baseline[:overlap_ms] > 0.0
                delta = (baseline[:overlap_ms] - candidate[:overlap_ms]) * 100.0 / baseline[:overlap_ms]
                selected_deltas << delta
                if delta > 0.5
                  wins += 1
                elsif delta < -0.5
                  losses += 1
                else
                  ties += 1
                end
              end
            else
              policy_total += baseline[:overlap_ms]
            end
          end
          next if selected == 0 || baseline_total <= 0.0
          delta_total = (baseline_total - policy_total) * 100.0 / baseline_total
          worst_delta = selected_deltas.empty? ? 0.0 : selected_deltas.min
          max_loss = [0.0, -worst_delta].max
          score = delta_total + worst_delta * 0.5 - losses * 10.0 - max_loss * 0.25
          policies << {
            key:            key,
            feature:        feature,
            op:             op,
            threshold:      threshold,
            prompts:        candidate_baselines.size,
            selected:       selected,
            wins:           wins,
            losses:         losses,
            ties:           ties,
            baseline_total: baseline_total,
            policy_total:   policy_total,
            delta:          delta_total,
            worst_delta:    worst_delta,
            max_loss:       max_loss,
            score:          score,
          }
        end
      end
    end
  end

  return if policies.empty?
  puts "self_spec_route_selector_scoreboard policies=#{policies.size} baselines=#{baselines.size} limit=#{limit}"
  puts "rank mode split route updown feature op threshold prompts selected wins losses ties baseline_total policy_total delta% worst_delta% max_loss% score"
  policies.sort { |a, b| b[:score] <=> a[:score] }.first(limit).each_with_index do |row, i|
    mode, split, route, updown = row[:key].split('|')
    puts "#{i + 1} #{mode} #{split} #{route} #{updown} #{row[:feature]} #{row[:op]} #{row[:threshold].round(4)} #{row[:prompts]} #{row[:selected]} #{row[:wins]} #{row[:losses]} #{row[:ties]} #{row[:baseline_total].round(3)} #{row[:policy_total].round(3)} #{row[:delta].round(2)} #{row[:worst_delta].round(2)} #{row[:max_loss].round(2)} #{row[:score].round(4)}"
  end
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
mtp_path = ENV["QWEN35_MTP"]? || DEFAULT_MTP
tokenizer_bin = ENV["LLAMA_TOKENIZE_BIN"]? || DEFAULT_TOKENIZER
prompt = DEFAULT_PROMPT
main_prompt_name = safe_prompt_label(ENV["QWEN35_PROMPT_NAME"]? || "main", "main")
prompt_as_prefix = false
tokens_limit = 96
calib_tokens = 32
layer_index = 0
ranks = [8, 16, 32, 64, 96, 128]
thresholds = [0.05, 0.10, 0.20, 0.35, 0.50]
basis_mode = "greedy"
pca_iters = 24
simulate_delta = false
simulate_dn_regime_features = false
dn_regime_g_cuts = [0.50, 0.75, 0.90, 0.95, 0.98]
simulate_lowrank = false
simulate_lowrank_metal = false
simulate_lowrank_metal_project = false
simulate_lowrank_metal_chunk = false
simulate_lowrank_metal_chunk_out = false
simulate_lowrank_metal_layer_chunk = false
simulate_lowrank_metal_layer_full = false
simulate_lowrank_metal_layer_updown_rank : Int32? = nil
simulate_lowrank_metal_layer_overlap = false
simulate_lowrank_metal_verifier_overlap = false
simulate_lowrank_metal_decode_verifier_overlap = false
simulate_exact_verifier_ltp = false
simulate_cost_truth_chunks = [] of Int32
simulate_cost_truth_branch_split_guards = [] of Int32
simulate_cost_truth_updown_rank : Int32? = nil
simulate_cost_truth_updown_layers = [] of Int32
simulate_current_hidden_proposal = false
simulate_current_hidden_proposal_topk = 5
simulate_current_hidden_transition_rank = 8
simulate_current_hidden_proposal_suite_prompts = [] of NamedPrompt
simulate_block_surrogate_start : Int32? = nil
simulate_block_surrogate_end : Int32? = nil
simulate_block_surrogate_rank : Int32? = nil
simulate_block_surrogate_clusters = 1
simulate_block_surrogate_policy = false
simulate_block_surrogate_state_mode = "skip"
simulate_block_surrogate_error_feedback_decays = [] of Float64
simulate_block_surrogate_delta_basis_modes = ["pca"]
simulate_block_surrogate_min_ideal_speedup = 1.0
simulate_block_surrogate_self_spec_gammas = [] of Int32
simulate_block_surrogate_suite_blocks = [] of LayerBlock
simulate_block_surrogate_suite_prompts = [] of NamedPrompt
simulate_block_surrogate_oracle_gen_calib = 0
simulate_block_surrogate_tree_oracle_k : Int32? = nil
simulate_block_surrogate_tree_warmup_tokens = 0
simulate_block_surrogate_tree_prefill_seed = false
simulate_block_surrogate_tree_branch_verify = false
simulate_block_surrogate_tree_select_advance = false
simulate_block_surrogate_topk_oracle_k : Int32? = nil
simulate_block_surrogate_topk_oracle_train_tokens : Int32? = nil
simulate_lowrank_metal_chunk_thread_overlap = false
simulate_multilayer_overlap_n = 0
simulate_logit_rank : Int32? = nil
simulate_logit_layers = [] of Int32
simulate_lowrank_eval_suite = false
simulate_lowrank_eval_suite_prompts = [] of NamedPrompt
simulate_fallback_threshold : Float64? = nil
simulate_fallback_thresholds = [] of Float64
simulate_generate_tokens = 0
simulate_output_margin_threshold : Float64? = nil
simulate_refresh_interval : Int32? = nil
simulate_oracle_refresh_interval : Int32? = nil
simulate_self_spec_gammas = [] of Int32
simulate_self_spec_adaptive = false
simulate_self_spec_adaptive_min = 4
simulate_self_spec_adaptive_start = 4
simulate_self_spec_adaptive_max = 16
simulate_self_spec_adaptive_grow_margin : Float64? = nil
simulate_self_spec_draft_margin : Float64? = nil
simulate_self_spec_draft_stop_margin : Float64? = nil
simulate_self_spec_topk_rescue : Int32? = nil
simulate_self_spec_tree_k : Int32? = nil
simulate_topk_oracle_k : Int32? = nil
simulate_topk_oracle_train_tokens : Int32? = nil
simulate_self_spec_progressive = [] of Int32
simulate_self_spec_wall_progressive = [] of Int32
simulate_cheap_self_draft_variants = [] of String
ffn_pca_calib_prompts = [] of String
simulate_ffn_block_sparsity_layers = [] of Int32
simulate_ffn_block_selector_layers = [] of Int32
simulate_ffn_block_selector_percents = [10, 20]
simulate_ffn_block_size = 256
simulate_self_spec_gpu_pipeline_suite_prompts = [] of NamedTuple(name: String, text: String)
simulate_ffn_updown_metal_rank : Int32? = nil
simulate_self_spec_wall_metal_lowrank = false
simulate_self_spec_wall_metal_project = false
simulate_self_spec_wall_metal_layer_updown = false
simulate_self_draft_metal_baseline = 0
simulate_self_draft_gpu_chain = 0
simulate_self_draft_gpu_chain_text = false
simulate_self_draft_gpu_chain_updown_rank : Int32? = nil
simulate_self_draft_gpu_chain_updown_layers = [] of Int32
simulate_self_draft_gpu_state_only = 0
simulate_self_draft_gpu_chain_overlap = 0
simulate_mtp_self_draft_fusion = 0
simulate_mtp_self_draft_fusion_topk = 5
simulate_mtp_self_draft_fusion_updown_rank : Int32? = nil
simulate_mtp_self_draft_fusion_updown_layers = [] of Int32
simulate_mtp_self_draft_fusion_suite_prompts = [] of NamedTuple(name: String, text: String)
simulate_self_spec_gpu_pipeline = 0
simulate_self_spec_gpu_pipeline_gammas = [] of Int32
simulate_self_spec_gpu_pipeline_schedules = [] of Array(Int32)
simulate_self_spec_gpu_pipeline_draft_splits = [] of Int32
simulate_self_spec_gpu_pipeline_dump_cycles_path : String? = ENV["QWEN35_SELF_SPEC_GPU_PIPELINE_DUMP_CYCLES"]?
simulate_self_spec_gpu_pipeline_no_backup = false
simulate_self_spec_gpu_pipeline_draft_no_ffn = false
simulate_self_spec_gpu_pipeline_draft_no_ffn_layers = [] of Int32
simulate_self_spec_gpu_pipeline_draft_no_ffn_fallback_abba = 0
simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn = false
simulate_self_spec_gpu_pipeline_draft_updown_rank : Int32? = nil
simulate_self_spec_gpu_pipeline_draft_updown_ranks = [] of Int32
simulate_self_spec_gpu_pipeline_draft_updown_repeats = 1
simulate_self_spec_gpu_pipeline_draft_updown_layers = [] of Int32
simulate_self_spec_gpu_pipeline_draft_updown_categories = [] of String
simulate_self_spec_gpu_pipeline_draft_updown_route_memory_root : String? = nil
simulate_self_spec_gpu_pipeline_draft_updown_route_key : String? = nil
simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject = false
simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts = 0
simulate_self_spec_gpu_pipeline_draft_updown_min_margin : Float64? = nil
simulate_self_spec_gpu_pipeline_draft_updown_max_chunks : Int32? = nil
simulate_self_spec_gpu_pipeline_draft_updown_after_rejects = 0
simulate_self_spec_gpu_pipeline_draft_updown_race_first_chunk = false
simulate_self_spec_gpu_pipeline_draft_updown_first_margin_threshold : Float64? = nil
simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept = false
simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate = false
simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps = 1
simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds = [] of Float64
simulate_self_spec_gpu_pipeline_legacy_full_state_backup = false
simulate_self_spec_gpu_pipeline_tree2_first = false
simulate_self_spec_gpu_pipeline_tree2_anywhere = false
simulate_self_spec_gpu_pipeline_tree2_staged_tokens = 0
simulate_self_spec_gpu_pipeline_tree2_margin_guard : Float64? = nil
simulate_self_spec_gpu_pipeline_tree2_branch_guard : Float64? = nil
simulate_self_spec_gpu_pipeline_branch_snapshot_policy : String? = nil
simulate_self_spec_gpu_pipeline_branch_snapshot_modes = [] of String
simulate_self_spec_gpu_pipeline_risk_offramp_margin : Float64? = nil
simulate_self_spec_gpu_pipeline_risk_offramp_margins = [] of Float64
simulate_self_spec_gpu_pipeline_risk_offramp_repeats = 1
simulate_self_spec_gpu_pipeline_mtp_k2_on_reject = false
simulate_self_spec_gpu_pipeline_reject_offramp_after = 0
simulate_self_spec_gpu_pipeline_attribution = ENV["QWEN35_SELF_SPEC_ATTR"]? == "1"
simulate_self_spec_gpu_pipeline_hybrid_sweep = false
simulate_self_spec_gpu_pipeline_hybrid_rich_sweep = false
simulate_self_spec_gpu_pipeline_suite_hybrid_sweep = false
simulate_self_spec_gpu_pipeline_route_features = false
simulate_self_spec_gpu_pipeline_ffn_updown_route_features = false
simulate_self_spec_gpu_pipeline_ffn_updown_hadamard_quant_features = false
simulate_self_spec_gpu_pipeline_ffn_updown_hadamard_bits = [8, 4]
simulate_self_spec_gpu_pipeline_ffn_updown_hadamard_blocks = [16, 32, 64]
ffn_updown_adapter_quant_bits : Int32? = nil
ffn_updown_adapter_quant_hadamard_block : Int32? = nil
ffn_updown_adapter_q8_metal = false
dump_ffn_updown_adapters_path : String? = nil
simulate_self_spec_gpu_pipeline_route_scoreboard = false
simulate_self_spec_gpu_pipeline_router_trace_path : String? = nil
simulate_self_spec_gpu_pipeline_route_selector_route : String? = nil
simulate_self_spec_gpu_pipeline_route_selector_no_ffn_layers = [] of Int32
simulate_self_spec_gpu_pipeline_route_selector_feature : String? = nil
simulate_self_spec_gpu_pipeline_route_selector_op = ">="
simulate_self_spec_gpu_pipeline_route_selector_threshold : Float64? = nil
simulate_self_spec_gpu_pipeline_route_selector_abba = 0
simulate_self_spec_gpu_pipeline_residual_router_mean_max : Float64? = nil
simulate_self_spec_gpu_pipeline_residual_router_pass_threshold : Float64? = nil
simulate_self_spec_gpu_pipeline_residual_router_pass_rate_min : Float64? = nil
simulate_self_spec_gpu_pipeline_value_repeat_rate_min : Float64? = nil
simulate_self_spec_gpu_pipeline_value_bigram_repeat_rate_min : Float64? = nil
simulate_self_spec_gpu_pipeline_value_unique_rate_max : Float64? = nil
self_spec_cost_model = false
self_spec_draft_cost = 0.0
self_spec_verifier_cost = 0.0
self_spec_chunk_overhead = 0.0
self_spec_correction_cost = 1.0
self_spec_overlap_cost = false
self_spec_overlap_efficiency = 1.0

add_self_spec_suite_prompt = ->(raw : String) {
  if sep = raw.index("::")
    name = raw[0, sep]
    text = raw[(sep + 2)..]
  else
    name = "suite#{simulate_self_spec_gpu_pipeline_suite_prompts.size + 1}"
    text = raw
  end
  safe_name = name.empty? ? "suite#{simulate_self_spec_gpu_pipeline_suite_prompts.size + 1}" : name.gsub(/[^A-Za-z0-9_.-]/, "_")
  simulate_self_spec_gpu_pipeline_suite_prompts << {name: safe_name, text: text}
}

add_current_hidden_proposal_suite_prompt = ->(raw : String) {
  if sep = raw.index("::")
    name = raw[0, sep]
    text = raw[(sep + 2)..]
  else
    name = "suite#{simulate_current_hidden_proposal_suite_prompts.size + 1}"
    text = raw
  end
  safe_name = name.empty? ? "suite#{simulate_current_hidden_proposal_suite_prompts.size + 1}" : name.gsub(/[^A-Za-z0-9_.-]/, "_")
  simulate_current_hidden_proposal_suite_prompts << {name: safe_name, text: text}
  simulate_current_hidden_proposal = true
}

add_mtp_self_draft_fusion_suite_prompt = ->(raw : String) {
  if sep = raw.index("::")
    name = raw[0, sep]
    text = raw[(sep + 2)..]
  else
    name = "suite#{simulate_mtp_self_draft_fusion_suite_prompts.size + 1}"
    text = raw
  end
  safe_name = name.empty? ? "suite#{simulate_mtp_self_draft_fusion_suite_prompts.size + 1}" : name.gsub(/[^A-Za-z0-9_.-]/, "_")
  simulate_mtp_self_draft_fusion_suite_prompts << {name: safe_name, text: text}
}

add_block_surrogate_suite_prompt = ->(raw : String) {
  if sep = raw.index("::")
    name = raw[0, sep]
    text = raw[(sep + 2)..]
  else
    name = "suite#{simulate_block_surrogate_suite_prompts.size + 1}"
    text = raw
  end
  safe_name = name.empty? ? "suite#{simulate_block_surrogate_suite_prompts.size + 1}" : name.gsub(/[^A-Za-z0-9_.-]/, "_")
  simulate_block_surrogate_suite_prompts << {name: safe_name, text: text}
}

add_lowrank_eval_suite_prompt = ->(raw : String) {
  if sep = raw.index("::")
    name = raw[0, sep]
    text = raw[(sep + 2)..]
  else
    name = "suite#{simulate_lowrank_eval_suite_prompts.size + 1}"
    text = raw
  end
  safe_name = name.empty? ? "suite#{simulate_lowrank_eval_suite_prompts.size + 1}" : name.gsub(/[^A-Za-z0-9_.-]/, "_")
  simulate_lowrank_eval_suite_prompts << {name: safe_name, text: text}
  simulate_lowrank_eval_suite = true
}

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_fixed_basis_probe [--model PATH] [--tokenizer PATH] [--prompt TEXT] [--tokens N] [--calib-tokens N] [--layer N] [--ranks LIST] [--basis greedy|pca]"
  p.on("--model=PATH", "GGUF model path") { |v| model = v }
  p.on("--mtp=PATH", "Qwen3.6 MTP safetensors sidecar path for MTP/self-draft fusion probes") { |v| mtp_path = v }
  p.on("--tokenizer=PATH", "llama-tokenize path") { |v| tokenizer_bin = v }
  p.on("--prompt=TEXT", "Prompt text") { |v| prompt = v }
  p.on("--prompt-as-prefix", "Use --prompt tokens once as the generation prefix instead of repeating it to fill --tokens") { prompt_as_prefix = true }
  p.on("--prompt-name=NAME", "Stable label for the main --prompt in self-spec traces") { |v| main_prompt_name = safe_prompt_label(v, "main") }
  p.on("--tokens=N", "Max prompt tokens to use") { |v| tokens_limit = v.to_i }
  p.on("--calib-tokens=N", "Tokens used to build the fixed basis") { |v| calib_tokens = v.to_i }
  p.on("--layer=N", "Recurrent layer index to probe (default: 0)") { |v| layer_index = v.to_i }
  p.on("--ranks=LIST", "Comma-separated ranks") { |v| ranks = v.split(',').map(&.to_i) }
  p.on("--thresholds=LIST", "Comma-separated residual thresholds for pass-rate reporting") { |v| thresholds = v.split(',').map(&.to_f) }
  p.on("--basis=MODE", "Basis builder: greedy or pca (default: greedy)") { |v| basis_mode = v }
  p.on("--pca-iters=N", "Power iterations per PCA component (default: 24)") { |v| pca_iters = v.to_i }
  p.on("--simulate-delta", "Also simulate projected-K DeltaNet output/state drift") { simulate_delta = true }
  p.on("--simulate-dn-regime-features", "Report g/beta/decay-horizon vs K residual features for projected-K routing") { simulate_dn_regime_features = true }
  p.on("--dn-regime-g-cuts=LIST", "Comma-separated g cutoffs for --simulate-dn-regime-features") { |v| dn_regime_g_cuts = parse_float_list(v) }
  p.on("--simulate-lowrank", "Also prove low-rank M*B^T recurrence against full projected-K recurrence") { simulate_lowrank = true }
  p.on("--simulate-lowrank-metal", "Compare Metal low-rank DeltaNet step against the CPU low-rank proof kernel") { simulate_lowrank_metal = true }
  p.on("--simulate-lowrank-metal-project", "Compare Metal Q/K projection plus low-rank DeltaNet step against the CPU proof kernel") { simulate_lowrank_metal_project = true }
  p.on("--simulate-lowrank-metal-chunk", "Compare one-command-buffer Metal low-rank chunk scan against CPU low-rank steps") { simulate_lowrank_metal_chunk = true }
  p.on("--simulate-lowrank-metal-chunk-out", "Compare fused Metal low-rank chunk scan+postnorm+ssm_out against CPU steps") { simulate_lowrank_metal_chunk_out = true }
  p.on("--simulate-lowrank-metal-layer-chunk", "Compare fused Metal low-rank recurrent attention chunk plus CPU FFN against CPU low-rank layer steps") { simulate_lowrank_metal_layer_chunk = true }
  p.on("--simulate-lowrank-metal-layer-full", "Compare one-command-buffer Metal low-rank recurrent layer chunk against CPU low-rank layer steps") { simulate_lowrank_metal_layer_full = true }
  p.on("--simulate-lowrank-metal-layer-updown=R", "Compare integrated Metal low-rank recurrent layer chunk with FFN pca-updown rank R against CPU pca-updown") { |v| simulate_lowrank_metal_layer_updown_rank = v.to_i }
  p.on("--simulate-lowrank-metal-layer-overlap", "Compare serial vs queued async full low-rank layer chunk submissions") { simulate_lowrank_metal_layer_overlap = true }
  p.on("--simulate-lowrank-metal-verifier-overlap", "Overlap one async low-rank layer chunk with exact prefill verifier on the held-out span") { simulate_lowrank_metal_verifier_overlap = true }
  p.on("--simulate-lowrank-metal-decode-verifier-overlap", "Overlap one async low-rank layer chunk with queued exact decode-wave verifier on the held-out span") { simulate_lowrank_metal_decode_verifier_overlap = true }
  p.on("--simulate-exact-verifier-ltp", "Compare exact verifier routes: serial decode, queued decode, and chunk-major prefill") { simulate_exact_verifier_ltp = true }
  p.on("--simulate-cost-truth-table=LIST", "Print normalized cost table for exact decode, chunk verifier, low-rank draft, and optional pca-updown draft over chunk sizes") { |v| simulate_cost_truth_chunks = parse_int_list(v) }
  p.on("--simulate-cost-truth-branch-splits=LIST", "With --simulate-cost-truth-table, also measure branch-guard verifier split shapes at 0-based guard indices; use -1 for all guards") { |v| simulate_cost_truth_branch_split_guards = parse_int_list(v) }
  p.on("--simulate-cost-truth-updown=R", "Include resident FFN pca-updown rank R in --simulate-cost-truth-table") { |v| simulate_cost_truth_updown_rank = v.to_i }
  p.on("--simulate-cost-truth-updown-layers=LIST", "Apply pca-updown cost-table rows only to the listed low-rank recurrent draft layers") { |v| simulate_cost_truth_updown_layers = parse_int_list(v) }
  p.on("--simulate-current-hidden-proposal", "Probe a verifier-safe proposal cache: nearest current final-hidden row -> exact top1 label") { simulate_current_hidden_proposal = true }
  p.on("--current-hidden-proposal-topk=N", "Candidate width for --simulate-current-hidden-proposal nearest-label coverage (default: 5)") { |v| simulate_current_hidden_proposal_topk = v.to_i }
  p.on("--current-hidden-transition-rank=N", "Rank for the PCA hidden-delta transition inside --simulate-current-hidden-proposal (default: 8)") { |v| simulate_current_hidden_transition_rank = v.to_i }
  p.on("--current-hidden-proposal-suite-prompt=NAME::TEXT", "Additional prompt for --simulate-current-hidden-proposal; main --prompt is always included") do |v|
    add_current_hidden_proposal_suite_prompt.call(v)
  end
  p.on("--current-hidden-proposal-suite-prompts-file=PATH", "Read current-hidden proposal suite prompts from UTF-8 lines: NAME::TEXT or TEXT") do |path|
    File.each_line(path) do |line|
      raw = line.strip
      next if raw.empty? || raw.starts_with?("#")
      add_current_hidden_proposal_suite_prompt.call(raw)
    end
  end
  p.on("--simulate-block-residual-surrogate=START:END", "Probe a static low-rank residual surrogate for a contiguous layer block on exact teacher-forced trajectory") do |v|
    block = parse_layer_block(v)
    simulate_block_surrogate_start = block[:start]
    simulate_block_surrogate_end = block[:end]
  end
  p.on("--block-surrogate-rank=N", "Rank for --simulate-block-residual-surrogate; defaults to --simulate-logits-rank or max --ranks") { |v| simulate_block_surrogate_rank = v.to_i }
  p.on("--block-surrogate-clusters=N", "Train N local residual adapters selected by nearest input centroid for --simulate-block-residual-surrogate") { |v| simulate_block_surrogate_clusters = v.to_i }
  p.on("--simulate-block-surrogate-policy", "Also substitute the trained block surrogate into the full model and report logit/greedy top-k drift") { simulate_block_surrogate_policy = true }
  p.on("--block-surrogate-state-mode=MODE", "State handling for block policy: skip (cheap/stateless) or shadow (exact state update, surrogate output)") { |v| simulate_block_surrogate_state_mode = v }
  p.on("--block-surrogate-error-feedback=LIST", "Probe one-token-lag EWMA residual-error correction decays for block residual predictions, e.g. 0,0.5,0.9") { |v| simulate_block_surrogate_error_feedback_decays = parse_float_list(v) }
  p.on("--block-surrogate-delta-basis=LIST", "Comma-separated delta bases for global block surrogate: pca,impact,balanced") { |v| simulate_block_surrogate_delta_basis_modes = v.split(',').map(&.strip).reject(&.empty?) }
  p.on("--block-surrogate-min-ideal-speedup=F", "Fail-closed economics marker for block-surrogate self-spec rows; default 1.0 means ideal overlap must beat paired exact") { |v| simulate_block_surrogate_min_ideal_speedup = v.to_f }
  p.on("--block-surrogate-oracle-gen-calib=N", "Probe-only upper bound: add N exact generated-token block samples to training while still drafting from the original prompt boundary") { |v| simulate_block_surrogate_oracle_gen_calib = v.to_i }
  p.on("--simulate-block-surrogate-self-spec-gammas=LIST", "Run exact self-spec acceptance gate for block-surrogate draft proposals at comma-separated gammas") { |v| simulate_block_surrogate_self_spec_gammas = parse_int_list(v) }
  p.on("--simulate-block-surrogate-tree-oracle=K", "Run block-surrogate top-K tree oracle using --simulate-block-surrogate-self-spec-gammas as fixed chunk schedules") { |v| simulate_block_surrogate_tree_oracle_k = v.to_i }
  p.on("--simulate-block-surrogate-tree-warmup=N", "Decode the first N generated tokens exactly before enabling block-surrogate tree oracle") { |v| simulate_block_surrogate_tree_warmup_tokens = v.to_i }
  p.on("--simulate-block-surrogate-prefill-seed", "Treat the first generated token as coming from exact final prompt logits before enabling block-surrogate tree oracle") { simulate_block_surrogate_tree_prefill_seed = true }
  p.on("--simulate-block-surrogate-tree-branch-verify", "Actually fork and advance exact branch states rank-order for block-surrogate tree candidates, instead of only scoring oracle coverage") { simulate_block_surrogate_tree_branch_verify = true }
  p.on("--simulate-block-surrogate-tree-select-advance", "Advance only the exact selected draft top-K branch state; lower-bound cost probe for mismatch-only rescue") { simulate_block_surrogate_tree_select_advance = true }
  p.on("--simulate-block-surrogate-topk-oracle=K", "Train/test a lightweight token/rank-bias reranker inside block-surrogate draft top-K") { |v| simulate_block_surrogate_topk_oracle_k = v.to_i }
  p.on("--simulate-block-surrogate-topk-oracle-train-tokens=N", "Training samples from the start of --simulate-generate for --simulate-block-surrogate-topk-oracle") { |v| simulate_block_surrogate_topk_oracle_train_tokens = v.to_i }
  p.on("--simulate-block-surrogate-suite-blocks=LIST", "Run one-process block-surrogate suite over blocks; use 24-30 for singletons or START:END items") { |v| simulate_block_surrogate_suite_blocks = parse_layer_block_list(v) }
  p.on("--simulate-block-surrogate-suite-prompt=NAME::TEXT", "Additional prompt for block-surrogate suite; main --prompt is always included") do |v|
    add_block_surrogate_suite_prompt.call(v)
  end
  p.on("--simulate-block-surrogate-suite-prompts-file=PATH", "Read block-surrogate suite prompts from UTF-8 lines: NAME::TEXT or TEXT") do |path|
    File.each_line(path) do |line|
      raw = line.strip
      next if raw.empty? || raw.starts_with?("#")
      add_block_surrogate_suite_prompt.call(raw)
    end
  end
  p.on("--simulate-lowrank-metal-chunk-thread-overlap", "Overlap one async low-rank layer chunk with chunk-major verifier in a worker thread") { simulate_lowrank_metal_chunk_thread_overlap = true }
  p.on("--simulate-lowrank-multilayer-chunk-thread-overlap=N", "Overlap N chained async low-rank layer chunks on one lane queue with chunk-major verifier in a worker thread") { |v| simulate_multilayer_overlap_n = v.to_i }
  p.on("--simulate-logits-rank=N", "Run full-model logit drift gate for one rank") { |v| simulate_logit_rank = v.to_i }
  p.on("--simulate-logits-layers=LIST", "Comma-separated recurrent layers to approximate together during the logit drift gate") { |v| simulate_logit_layers = parse_int_list(v) }
  p.on("--simulate-lowrank-eval-suite", "Run paired low-rank DeltaNet logit/greedy/self-spec eval for main plus suite prompts") { simulate_lowrank_eval_suite = true }
  p.on("--simulate-lowrank-eval-suite-prompt=NAME::TEXT", "Additional prompt for --simulate-lowrank-eval-suite; main --prompt is always included") do |v|
    add_lowrank_eval_suite_prompt.call(v)
  end
  p.on("--simulate-lowrank-eval-suite-prompts-file=PATH", "Read low-rank eval suite prompts from UTF-8 lines: NAME::TEXT or TEXT") do |path|
    File.each_line(path) do |line|
      raw = line.strip
      next if raw.empty? || raw.starts_with?("#")
      add_lowrank_eval_suite_prompt.call(raw)
    end
  end
  p.on("--simulate-fallback-threshold=F", "Fallback to exact DeltaNet step when max per-head K residual exceeds F") { |v| simulate_fallback_threshold = v.to_f64 }
  p.on("--simulate-fallback-thresholds=LIST", "Run multiple fallback thresholds in one process") { |v| simulate_fallback_thresholds = v.split(',').map(&.strip).reject(&.empty?).map(&.to_f64) }
  p.on("--simulate-fallback-score=MODE", "Fallback score for threshold routing: raw, decayed (g*residual), or update (beta*residual)") { |v| ProbeRuntime.fallback_score_mode = v }
  p.on("--simulate-output-margin-threshold=F", "Fallback to exact output when approximate top1/top2 margin is below F") { |v| simulate_output_margin_threshold = v.to_f64 }
  p.on("--simulate-refresh-interval=N", "Force an exact low-rank-state refresh every N approximate-eligible positions") { |v| simulate_refresh_interval = v.to_i }
  p.on("--simulate-oracle-refresh-interval=N", "Copy the paired exact shadow state into the approximate state every N positions") { |v| simulate_oracle_refresh_interval = v.to_i }
  p.on("--simulate-generate=N", "Run teacher-forced exact-greedy generation drift gate for N decode tokens") { |v| simulate_generate_tokens = v.to_i }
  p.on("--simulate-self-spec-gammas=LIST", "Run self-spec low-rank draft simulation for comma-separated gammas") { |v| simulate_self_spec_gammas = parse_int_list(v) }
  p.on("--simulate-self-spec-adaptive=MIN,START,MAX", "Run adaptive self-spec gamma: grow on full accept, shrink on reject") do |v|
    values = parse_int_list(v)
    raise "adaptive self-spec expects MIN,START,MAX" unless values.size == 3
    simulate_self_spec_adaptive = true
    simulate_self_spec_adaptive_min = values[0]
    simulate_self_spec_adaptive_start = values[1]
    simulate_self_spec_adaptive_max = values[2]
  end
  p.on("--simulate-self-spec-adaptive-grow-margin=F", "Only grow adaptive self-spec gamma when exact verifier min margin in the chunk is at least F") { |v| simulate_self_spec_adaptive_grow_margin = v.to_f64 }
  p.on("--simulate-self-spec-draft-margin=F", "Count low-margin draft proposal steps below F inside each self-spec chunk") { |v| simulate_self_spec_draft_margin = v.to_f64 }
  p.on("--simulate-self-spec-draft-stop-margin=F", "Stop a self-spec proposal chunk once draft margin falls below F after the min gamma") { |v| simulate_self_spec_draft_stop_margin = v.to_f64 }
  p.on("--simulate-self-spec-topk-rescue=K", "Treat a greedy reject as tree-rescued when exact token is in draft top-K") { |v| simulate_self_spec_topk_rescue = v.to_i }
  p.on("--simulate-self-spec-tree-k=K", "Run progressive top-K tree oracle using --simulate-self-spec-progressive as the schedule") { |v| simulate_self_spec_tree_k = v.to_i }
  p.on("--simulate-topk-oracle=K", "Train/test a lightweight token/rank-bias reranker inside low-rank draft top-K") { |v| simulate_topk_oracle_k = v.to_i }
  p.on("--simulate-topk-oracle-train-tokens=N", "Training samples from the start of --simulate-generate for --simulate-topk-oracle") { |v| simulate_topk_oracle_train_tokens = v.to_i }
  p.on("--simulate-self-spec-progressive=LIST", "Run progressive self-spec verifier chunks with a repeating comma-separated schedule, e.g. 4,4,8") { |v| simulate_self_spec_progressive = parse_int_list(v) }
  p.on("--simulate-self-spec-wall-progressive=LIST", "Measure wall-clock low-rank draft plus exact chunk verifier for a progressive schedule") { |v| simulate_self_spec_wall_progressive = parse_int_list(v) }
  p.on("--simulate-cheap-self-draft-variants=LIST", "Run wall self-spec with comma-separated draft variants: lowrank,lowrank-no-ffn,skip-layer,early-exit-N,lowrank-ffn-top-P,lowrank-ffn-blocktop-P,lowrank-ffn-blockpred-P,lowrank-ffn-pca-R,lowrank-ffn-pca-down-R,lowrank-ffn-pca-updown-R") { |v| simulate_cheap_self_draft_variants = v.split(',').map(&.strip).reject(&.empty?) }
  p.on("--ffn-pca-calib-prompt=TEXT", "Additional prompt used to build FFN PCA/PCA-down basis; may be repeated") { |v| ffn_pca_calib_prompts << v }
  p.on("--simulate-ffn-block-sparsity=LIST", "Measure how many FFN-down quant blocks retain SwiGLU activation energy for listed recurrent layers") { |v| simulate_ffn_block_sparsity_layers = parse_int_list(v) }
  p.on("--simulate-ffn-block-selector=LIST", "Nearest-neighbor probe: predict top FFN activation blocks from ffn_in for listed recurrent layers") { |v| simulate_ffn_block_selector_layers = parse_int_list(v) }
  p.on("--ffn-block-selector-percents=LIST", "Top-block percentages for --simulate-ffn-block-selector (default: 10,20)") { |v| simulate_ffn_block_selector_percents = parse_int_list(v) }
  p.on("--ffn-block-size=N", "Activation channels per FFN-down sparse block for --simulate-ffn-block-sparsity (default: 256)") { |v| simulate_ffn_block_size = v.to_i }
  p.on("--simulate-ffn-updown-metal=R", "Run Metal microkernel gate for FFN pca-updown rank R") { |v| simulate_ffn_updown_metal_rank = v.to_i }
  p.on("--simulate-self-spec-wall-metal-lowrank", "Use the Metal low-rank DeltaNet core inside wall-clock self-spec draft proposals") { simulate_self_spec_wall_metal_lowrank = true }
  p.on("--simulate-self-spec-wall-metal-project", "Also compute Q/K low-rank coefficients on Metal before the Metal low-rank step") { simulate_self_spec_wall_metal_lowrank = true; simulate_self_spec_wall_metal_project = true }
  p.on("--simulate-self-spec-wall-metal-layer-updown", "Route lowrank-ffn-pca-updown-R draft layers through the integrated Metal layer-updown path") { simulate_self_spec_wall_metal_lowrank = true; simulate_self_spec_wall_metal_project = true; simulate_self_spec_wall_metal_layer_updown = true }
  p.on("--simulate-self-draft-metal-baseline=N", "Wall-clock the Metal-only self-draft (low-rank on --simulate-logits-layers) vs exact greedy and chunk-major verifier on N held-out tokens") { |v| simulate_self_draft_metal_baseline = v.to_i }
  p.on("--simulate-self-draft-gpu-chain=N", "Queue N low-rank self-draft top1 steps with GPU top1_id -> next embedding and no intermediate CPU readback") { |v| simulate_self_draft_gpu_chain = v.to_i }
  p.on("--simulate-self-draft-gpu-chain-text", "With --simulate-self-draft-gpu-chain, decode draft/exact ids as escaped text for no-validator inspection") { simulate_self_draft_gpu_chain_text = true }
  p.on("--simulate-self-draft-gpu-chain-updown=R", "With --simulate-self-draft-gpu-chain, also run resident FFN pca-updown rank R for no-validator drift/text inspection") { |v| simulate_self_draft_gpu_chain_updown_rank = v.to_i }
  p.on("--simulate-self-draft-gpu-chain-updown-layers=LIST", "Apply self-draft gpu chain pca-updown only to the listed low-rank recurrent draft layers") { |v| simulate_self_draft_gpu_chain_updown_layers = parse_int_list(v) }
  p.on("--simulate-self-draft-gpu-state-only=N", "Queue N known-token low-rank draft state updates without lm-head/top1; lower-bound ablation for draft head/control cost") { |v| simulate_self_draft_gpu_state_only = v.to_i }
  p.on("--simulate-self-draft-gpu-chain-overlap=N", "Run GPU self-draft chain on a lane queue while chunk-major verifier runs on the default queue") { |v| simulate_self_draft_gpu_chain_overlap = v.to_i }
  p.on("--simulate-mtp-self-draft-fusion=N", "Probe MTP top-K as a verifier-rescue/fusion source over N GPU self-draft steps") { |v| simulate_mtp_self_draft_fusion = v.to_i }
  p.on("--simulate-mtp-self-draft-fusion-topk=K", "MTP top-K width for --simulate-mtp-self-draft-fusion (default: 5)") { |v| simulate_mtp_self_draft_fusion_topk = v.to_i }
  p.on("--simulate-mtp-self-draft-fusion-updown=R", "Use resident FFN pca-updown rank R in the self-draft side of the MTP fusion probe") { |v| simulate_mtp_self_draft_fusion_updown_rank = v.to_i }
  p.on("--simulate-mtp-self-draft-fusion-updown-layers=LIST", "Apply MTP fusion pca-updown only to the listed low-rank recurrent draft layers") { |v| simulate_mtp_self_draft_fusion_updown_layers = parse_int_list(v) }
  p.on("--simulate-mtp-self-draft-fusion-suite-prompt=NAME::TEXT", "Additional eval prompt for MTP/self-draft fusion suite; main --prompt still runs first") do |v|
    add_mtp_self_draft_fusion_suite_prompt.call(v)
  end
  p.on("--simulate-mtp-self-draft-fusion-suite-prompts-file=PATH", "Read MTP/self-draft fusion suite prompts from UTF-8 lines: NAME::TEXT or TEXT") do |path|
    File.each_line(path) do |line|
      raw = line.strip
      next if raw.empty? || raw.starts_with?("#")
      add_mtp_self_draft_fusion_suite_prompt.call(raw)
    end
  end
  p.on("--simulate-self-spec-gpu-pipeline=N", "Run real fixed-gamma self-spec block pipeline: draft[k+1] on lane queue while verifier validates draft[k]") { |v| simulate_self_spec_gpu_pipeline = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-gammas=LIST", "Run real GPU self-spec pipeline for comma-separated fixed gammas in one model load") { |v| simulate_self_spec_gpu_pipeline_gammas = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-schedule=LIST", "Run real GPU self-spec pipeline with a repeating gamma schedule that resets on reject, e.g. 4,4,8") { |v| simulate_self_spec_gpu_pipeline_schedules << parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-draft-splits=LIST", "Run real GPU self-spec pipeline with comma-separated draft command-buffer split sizes; 0 keeps one command buffer per draft block") { |v| simulate_self_spec_gpu_pipeline_draft_splits = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-dump-cycles=PATH", "Append atlas-compatible JSONL chunk records for real GPU self-spec pipeline runs") { |v| simulate_self_spec_gpu_pipeline_dump_cycles_path = v }
  p.on("--simulate-self-spec-gpu-pipeline-draft-exact-refresh=N", "Every Nth draft wave, reconstruct low-rank DN state to full, run exact recurrent DN for low-rank layers, then project back to low-rank; 0 disables") { |v| ProbeRuntime.gpu_draft_exact_refresh_interval = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-exact-refresh-prefix=N", "Exact-refresh the first N generated draft waves, then resume low-rank DN for later waves; 0 disables") { |v| ProbeRuntime.gpu_draft_exact_refresh_prefix = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-exact-refresh-offsets=LIST", "0-based generated-token offsets where draft waves should exact-refresh low-rank DN layers; default empty") { |v| ProbeRuntime.gpu_draft_exact_refresh_offsets = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-draft-update-risk-threshold=F", "Plan exact-refresh offsets from the CPU teacher-forced fallback score (raw/decayed/update via --simulate-fallback-score) without per-token GPU readback") { |v| ProbeRuntime.gpu_draft_update_risk_threshold = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-draft-update-risk-layer-threshold=F", "Plan per-layer exact-refresh offsets from the CPU teacher-forced fallback score without per-token GPU readback") { |v| ProbeRuntime.gpu_draft_update_risk_layer_threshold = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-no-backup", "Skip verifier rollback backup on the hot full-accept path; rebuild exact state from emitted ids on reject") { simulate_self_spec_gpu_pipeline_no_backup = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-no-ffn", "Use the research lowrank-no-ffn draft route for GPU self-spec proposals; exact verifier still enforces parity") { simulate_self_spec_gpu_pipeline_draft_no_ffn = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-no-ffn-layers=LIST", "Skip FFN only for the listed low-rank recurrent draft layers; enables hybrid draft bodies") { |v| simulate_self_spec_gpu_pipeline_draft_no_ffn_layers = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-draft-no-ffn-fallback-on-reject", "After the first no-FFN draft rejection, resync future draft blocks with baseline lowrank") { ProbeRuntime.self_spec_draft_no_ffn_fallback_on_reject = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-no-ffn-fallback-abba=N", "Run an in-process ABBA diagnostic comparing no-FFN fallback-on-reject off/on for N cycles") { |v| simulate_self_spec_gpu_pipeline_draft_no_ffn_fallback_abba = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-no-ffn-after-full-accepts=N", "Start no-FFN draft transport only after N consecutive full-accept chunks; rejects reset to pure") { |v| ProbeRuntime.self_spec_draft_no_ffn_after_full_accepts = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-no-ffn-min-margin=F", "Start/keep no-FFN draft transport only when the last full-accept chunk's min top1-top2 margin is at least F") { |v| ProbeRuntime.self_spec_draft_no_ffn_min_margin = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-draft-no-ffn-max-chunks=N", "Use no-FFN for at most N submitted draft chunks in one run; pairs with after-full-accepts/min-margin gates") { |v| ProbeRuntime.self_spec_draft_no_ffn_max_chunks = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-skip-recurrent-ffn", "Research route: skip FFN on all recurrent draft layers; exact verifier still enforces parity") { simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown=R", "Use resident FFN pca-updown rank R on selected low-rank recurrent draft layers in the real GPU pipeline") { |v| simulate_self_spec_gpu_pipeline_draft_updown_rank = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updowns=LIST", "In-process A/B list for resident FFN pca-updown ranks in the real GPU pipeline; use 0 for lowrank baseline") { |v| simulate_self_spec_gpu_pipeline_draft_updown_ranks = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-repeats=N", "Repeat draft body A/B in ABBA order and score against median lowrank baselines") { |v| simulate_self_spec_gpu_pipeline_draft_updown_repeats = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-layers=LIST", "Apply resident FFN pca-updown only to the listed low-rank recurrent draft layers; enables hybrid draft bodies") { |v| simulate_self_spec_gpu_pipeline_draft_updown_layers = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-categories=LIST", "Only enable pca-updown for prompt categories/names in LIST; prompt category is the prefix before '_'") { |v| simulate_self_spec_gpu_pipeline_draft_updown_categories = v.split(',').map(&.strip).reject(&.empty?) }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-route-memory=ROOT", "Lookup cached proposal-body route decisions before pca-updown adapter setup") { |v| simulate_self_spec_gpu_pipeline_draft_updown_route_memory_root = v }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-route-key=KEY", "With route memory, prefer this stable prompt/task route key over exact prompt-token lookup") { |v| simulate_self_spec_gpu_pipeline_draft_updown_route_key = v }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-fallback-on-reject", "After the first pca-updown draft rejection, resync future draft blocks with baseline lowrank") { simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-after-full-accepts=N", "Only enable pca-updown after N consecutive full-accept chunks; any reject disables and resets the streak") { |v| simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-min-margin=F", "Only enable pca-updown for future draft blocks after a full-accept chunk whose draft top2 min-margin is at least F; combines with --simulate-self-spec-gpu-pipeline-draft-updown-after-full-accepts") { |v| simulate_self_spec_gpu_pipeline_draft_updown_min_margin = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-max-chunks=N", "Use pca-updown for at most N draft chunks in a run; N=0 keeps the pca route closed and is useful for drift/burst falsifiers") { |v| simulate_self_spec_gpu_pipeline_draft_updown_max_chunks = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-after-rejects=N", "Start with baseline lowrank and enable pca-updown for future chunks after N exact verifier rejects; an updown reject disables it again") { |v| simulate_self_spec_gpu_pipeline_draft_updown_after_rejects = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-race-first-chunk", "Exact-score lowrank and pca-updown first chunks before committing; choose pca-updown only if it full-accepts while lowrank rejects") { simulate_self_spec_gpu_pipeline_draft_updown_race_first_chunk = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-first-margin=F", "Start with baseline lowrank; if the first seed chunk top1/top2 min-margin is <= F, switch that chunk and future chunks to pca-updown") { |v| simulate_self_spec_gpu_pipeline_draft_updown_first_margin_threshold = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-refresh-on-accept", "After a fully accepted pca-updown chunk, discard approximate draft state and seed the next chunk from exact verifier state") { simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-refresh-on-accept", "After every fully accepted draft chunk, discard approximate draft state and seed the next chunk from exact verifier state") { ProbeRuntime.self_spec_draft_refresh_on_accept = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-agreement-gate", "Probe lowrank vs pca-updown at the same boundary and use pca-updown only when its next top1 matches lowrank top1/top2") { simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate = true }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-agreement-steps=N", "Compare N lowrank/pca-updown draft steps before allowing pca-updown; steps after the first must match lowrank top1 exactly") { |v| simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-draft-updown-agreement-margin-thresholds=LIST", "When agreement gate is enabled, score lowrank same-boundary margin thresholds as candidate cheap pca-updown predictors; reports selected/pass/fail/false-negative counts") { |v| simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds = parse_float_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-residual-router-mean-max=F", "Default-off gate: run GPU self-spec only when held-out low-rank residual mean is <= F; otherwise emit exact fallback row") { |v| simulate_self_spec_gpu_pipeline_residual_router_mean_max = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-residual-router-pass-threshold=F", "Residual router pass-rate threshold paired with --simulate-self-spec-gpu-pipeline-residual-router-pass-rate-min") { |v| simulate_self_spec_gpu_pipeline_residual_router_pass_threshold = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-residual-router-pass-rate-min=F", "Default-off gate: require at least F percent of held-out residuals under the configured pass threshold") { |v| simulate_self_spec_gpu_pipeline_residual_router_pass_rate_min = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-value-repeat-rate-min=F", "Default-off value gate: require at least F percent repeated held-out prompt tokens before submitting GPU self-spec") { |v| simulate_self_spec_gpu_pipeline_value_repeat_rate_min = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-value-bigram-repeat-rate-min=F", "Default-off value gate: require at least F percent repeated held-out prompt bigrams before submitting GPU self-spec") { |v| simulate_self_spec_gpu_pipeline_value_bigram_repeat_rate_min = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-value-unique-rate-max=F", "Default-off value gate: require held-out prompt unique-token rate to be <= F percent before submitting GPU self-spec") { |v| simulate_self_spec_gpu_pipeline_value_unique_rate_max = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-legacy-full-state-backup", "Use the old full-capacity State#copy_from! verifier backup/restore path for A/B") { simulate_self_spec_gpu_pipeline_legacy_full_state_backup = true }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-first", "Use real draft top2 for a first-token k=2 early branch before verifying the wrong tail") { simulate_self_spec_gpu_pipeline_tree2_first = true }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-anywhere", "Use real draft top2 with serial exact verifier inside each draft chunk, stopping at the first mismatch") { simulate_self_spec_gpu_pipeline_tree2_anywhere = true }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-staged=N", "Use real draft top2 with chunk-major verifier stages of N tokens, stopping at the first mismatch") { |v| simulate_self_spec_gpu_pipeline_tree2_staged_tokens = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-margin-guard=F", "Use draft top1/top2 margin <= F to split exact verifier at the first low-margin token") { |v| simulate_self_spec_gpu_pipeline_tree2_margin_guard = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-branch-guard=F", "Use margin <= F to stop before the low-margin token and resync from exact branch state on reject") { |v| simulate_self_spec_gpu_pipeline_tree2_branch_guard = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-branch-guard-snapshot", "After a branch guard passes, snapshot the exact guard-boundary state so suffix rejects replay from the guard boundary") { ProbeRuntime.self_spec_branch_guard_snapshot = true }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-branch-guard-snapshot-min-prefix=N", "Snapshot branch-guard pass state only after at least N accepted guard-prefix tokens") { |v| ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-branch-guard-snapshot-suffix-min-threshold=F", "Only snapshot a passed branch guard if the remaining suffix contains a draft margin >= F and also passes the suffix threshold") { |v| ProbeRuntime.self_spec_branch_guard_snapshot_suffix_min_threshold = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-branch-guard-snapshot-suffix-threshold=F", "Only snapshot a passed branch guard if the remaining suffix contains a draft margin <= F") { |v| ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-branch-guard-snapshot-prefix-suffix-thresholds=LIST", "Prefix-conditioned snapshot suffix thresholds, e.g. 2:1.0,3:2.0; largest matching min-prefix wins and overrides the single suffix threshold") { |v| ProbeRuntime.self_spec_branch_guard_snapshot_prefix_suffix_thresholds = parse_prefix_suffix_thresholds(v) }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-branch-guard-until-reject", "Only enable branch guard before the first real reject in the run") { ProbeRuntime.self_spec_branch_guard_until_reject = true }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-branch-guard-overlap-next", "Allow speculative next-block pre-submit even when branch guard is active; rejects discard the wasted block") { ProbeRuntime.self_spec_branch_guard_overlap_next = true }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-branch-guard-snapshot-only-split", "When branch snapshots are enabled, split verifier only for guard candidates that pass the snapshot value gate") { ProbeRuntime.self_spec_branch_guard_snapshot_only_split = true }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-branch-guard-no-snapshot-threshold=F", "With snapshot-only split, still run no-snapshot branch guards for margins <= F") { |v| ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-tree2-branch-guard-single-pass-checkpoint", "When branch snapshots are enabled, verify prefix+guard+suffix in one known-span pass while checkpointing recurrent state after the guard") { ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = true }
  p.on("--simulate-self-spec-gpu-pipeline-branch-snapshot-policy=MODE", "Apply a calibrated branch snapshot policy: off, split, onepass, split_min3_suffix2, onepass_min3_suffix2, *_keepguard, or onepass guard005/01/02/05/inf") { |v| simulate_self_spec_gpu_pipeline_branch_snapshot_policy = apply_branch_snapshot_policy(v) }
  p.on("--simulate-self-spec-gpu-pipeline-branch-snapshot-modes=LIST", "In-process A/B for branch snapshot verifier modes: nosnap, split, split_min3_suffix2, onepass_min3_suffix2, *_keepguard, or onepass guard005/01/02/05/inf") { |v| simulate_self_spec_gpu_pipeline_branch_snapshot_modes = v.split(',').map(&.strip).reject(&.empty?) }
  p.on("--simulate-self-spec-gpu-pipeline-risk-offramp-margin=F", "When draft top1/top2 margin <= F, do not pre-submit the next draft block before exact verification") { |v| simulate_self_spec_gpu_pipeline_risk_offramp_margin = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-risk-offramp-margins=LIST", "In-process A/B list for risk-offramp thresholds; automatically includes the no-offramp baseline") { |v| simulate_self_spec_gpu_pipeline_risk_offramp_margins = parse_float_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-risk-offramp-repeats=N", "Repeat risk-offramp A/B in ABBA order and score against median no-offramp baselines") { |v| simulate_self_spec_gpu_pipeline_risk_offramp_repeats = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-mtp-k2-on-reject", "Diagnostic: on real pipeline self-top2 rejects, call Qwen3.6 MTP K2 from the exact boundary and count rescues without changing emitted tokens") { simulate_self_spec_gpu_pipeline_mtp_k2_on_reject = true }
  p.on("--simulate-self-spec-gpu-pipeline-reject-offramp-after=N", "After N self-spec rejects, stop drafting and finish the remaining requested tokens with exact greedy decode") { |v| simulate_self_spec_gpu_pipeline_reject_offramp_after = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-attribution", "Append WBA attribution counters for the real GPU self-spec pipeline") { simulate_self_spec_gpu_pipeline_attribution = true }
  p.on("--simulate-self-spec-gpu-pipeline-hybrid-sweep", "Run an in-process route sweep over pure/no-FFN/pca-updown hybrid layer masks") { simulate_self_spec_gpu_pipeline_hybrid_sweep = true }
  p.on("--simulate-self-spec-gpu-pipeline-hybrid-rich-sweep", "Add per-layer, prefix/suffix, and alternating hybrid routes to the GPU self-spec layer-mode sweep") { simulate_self_spec_gpu_pipeline_hybrid_sweep = true; simulate_self_spec_gpu_pipeline_hybrid_rich_sweep = true }
  p.on("--simulate-self-spec-gpu-pipeline-suite-hybrid-sweep", "Apply the hybrid route sweep to suite prompts and print aggregate prompt-stability ranking") { simulate_self_spec_gpu_pipeline_hybrid_sweep = true; simulate_self_spec_gpu_pipeline_suite_hybrid_sweep = true; simulate_self_spec_gpu_pipeline_route_features = true }
  p.on("--simulate-self-spec-gpu-pipeline-route-features", "Print held-out PCA residual features that can predict risky self-spec draft routes") { simulate_self_spec_gpu_pipeline_route_features = true }
  p.on("--simulate-self-spec-gpu-pipeline-ffn-updown-route-features", "Print held-out FFN pca-updown reconstruction residual features for prompt-level pca-updown route risk") { simulate_self_spec_gpu_pipeline_ffn_updown_route_features = true }
  p.on("--simulate-self-spec-gpu-pipeline-ffn-updown-hadamard-quant-features", "Print held-out FFN pca-updown adapter quantization probes: raw vs block-Hadamard q8/q4 coefficient/down-basis rows") { simulate_self_spec_gpu_pipeline_ffn_updown_hadamard_quant_features = true }
  p.on("--ffn-updown-hadamard-quant-bits=LIST", "Bits for --simulate-self-spec-gpu-pipeline-ffn-updown-hadamard-quant-features, default 8,4") { |v| simulate_self_spec_gpu_pipeline_ffn_updown_hadamard_bits = parse_int_list(v) }
  p.on("--ffn-updown-hadamard-blocks=LIST", "Power-of-two block sizes for Hadamard quant probes, default 16,32,64") { |v| simulate_self_spec_gpu_pipeline_ffn_updown_hadamard_blocks = parse_int_list(v) }
  p.on("--ffn-updown-adapter-quant-bits=N", "Quality proxy: replace trained FFN pca-updown adapter rows with symmetric qN-dequantized rows before draft simulation") { |v| ffn_updown_adapter_quant_bits = v.to_i }
  p.on("--ffn-updown-adapter-quant-hadamard-block=N", "With --ffn-updown-adapter-quant-bits, quantize adapter rows in block-Hadamard space before dequantizing back to f32") { |v| ffn_updown_adapter_quant_hadamard_block = v.to_i }
  p.on("--ffn-updown-adapter-q8-metal", "Use raw symmetric q8 adapter-row buffers in the Metal pca-updown draft kernel instead of f32 adapter rows") { ffn_updown_adapter_q8_metal = true }
  p.on("--dump-ffn-updown-adapters=PATH", "Write trained FFN pca-updown adapters as JSON for CUDA runner probes") { |v| dump_ffn_updown_adapters_path = v }
  p.on("--simulate-self-spec-gpu-pipeline-route-scoreboard", "Print a ranked route scoreboard after a GPU self-spec hybrid sweep") { simulate_self_spec_gpu_pipeline_route_scoreboard = true }
  p.on("--simulate-self-spec-gpu-pipeline-router-trace=PATH", "Write JSONL self-spec router/reject-risk rows without raw token ids") { |v| simulate_self_spec_gpu_pipeline_router_trace_path = v }
  p.on("--simulate-self-spec-gpu-pipeline-route-selector-route=NAME", "Default-off prompt route selector candidate, e.g. noffn_0 or noffn_0_2; pure is used when the feature gate does not fire") { |v| simulate_self_spec_gpu_pipeline_route_selector_route = v }
  p.on("--simulate-self-spec-gpu-pipeline-route-selector-no-ffn-layers=LIST", "Selector-local no-FFN layers for a custom route; does not affect the ordinary pure baseline row") { |v| simulate_self_spec_gpu_pipeline_route_selector_no_ffn_layers = parse_int_list(v) }
  p.on("--simulate-self-spec-gpu-pipeline-route-selector-feature=NAME", "Feature for route selector: residual_mean,residual_p90,residual_max,repeat_rate,bigram_repeat_rate,unique_rate") { |v| simulate_self_spec_gpu_pipeline_route_selector_feature = v }
  p.on("--simulate-self-spec-gpu-pipeline-route-selector-op=OP", "Route selector comparison: <= or >=") { |v| simulate_self_spec_gpu_pipeline_route_selector_op = v }
  p.on("--simulate-self-spec-gpu-pipeline-route-selector-threshold=F", "Threshold for --simulate-self-spec-gpu-pipeline-route-selector-feature") { |v| simulate_self_spec_gpu_pipeline_route_selector_threshold = v.to_f64 }
  p.on("--simulate-self-spec-gpu-pipeline-route-selector-abba=N", "Run route-selector pure/selector/selector/pure in-process ABBA cycles for N repeats") { |v| simulate_self_spec_gpu_pipeline_route_selector_abba = v.to_i }
  p.on("--simulate-self-spec-gpu-pipeline-suite-prompt=NAME::TEXT", "Additional eval prompt for GPU self-spec pipeline suite; main --prompt still runs first") do |v|
    add_self_spec_suite_prompt.call(v)
  end
  p.on("--simulate-self-spec-gpu-pipeline-suite-prompts-file=PATH", "Read additional suite prompts from a UTF-8 text file; each non-empty non-comment line is NAME::TEXT or TEXT") do |path|
    File.each_line(path) do |line|
      raw = line.strip
      next if raw.empty? || raw.starts_with?("#")
      add_self_spec_suite_prompt.call(raw)
    end
  end
  p.on("--self-spec-draft-cost=F", "Relative cost per low-rank draft token (plain exact decode token = 1)") { |v| self_spec_cost_model = true; self_spec_draft_cost = v.to_f64 }
  p.on("--self-spec-verifier-cost=F", "Relative cost per exact verifier token in a chunk (plain exact decode token = 1)") { |v| self_spec_cost_model = true; self_spec_verifier_cost = v.to_f64 }
  p.on("--self-spec-chunk-overhead=F", "Relative fixed overhead per self-spec chunk") { |v| self_spec_cost_model = true; self_spec_chunk_overhead = v.to_f64 }
  p.on("--self-spec-correction-cost=F", "Relative cost per rejected-token correction step") { |v| self_spec_cost_model = true; self_spec_correction_cost = v.to_f64 }
  p.on("--self-spec-overlap-cost", "Estimate draft/verifier pipeline cost as max(draft, verifier) per chunk") { self_spec_cost_model = true; self_spec_overlap_cost = true }
  p.on("--self-spec-overlap-efficiency=F", "Fraction of min(draft, verifier) hidden by overlap, 0..1 (default: 1)") { |v| self_spec_cost_model = true; self_spec_overlap_cost = true; self_spec_overlap_efficiency = v.to_f64 }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "model not found: #{model}" unless File.exists?(model)
raise "tokenizer not found: #{tokenizer_bin}" unless File.exists?(tokenizer_bin)
raise "tokens must be positive" unless tokens_limit > 0
raise "calib-tokens must be positive" unless calib_tokens > 0
raise "ranks must not be empty" if ranks.empty?
raise "pca-iters must be positive" unless pca_iters > 0
raise "current-hidden proposal topK must be positive" unless simulate_current_hidden_proposal_topk > 0
raise "current-hidden transition rank must be non-negative" unless simulate_current_hidden_transition_rank >= 0
raise "FFN block size must be positive" unless simulate_ffn_block_size > 0
raise "FFN block selector percentages must be in 1..100" if simulate_ffn_block_selector_percents.any? { |v| v < 1 || v > 100 }
pipeline_router_trace_io = simulate_self_spec_gpu_pipeline_router_trace_path.try { |path| File.open(path, "w") }
ProbeRuntime.self_spec_router_trace_io = pipeline_router_trace_io
ProbeRuntime.self_spec_draft_updown_race_first_chunk = simulate_self_spec_gpu_pipeline_draft_updown_race_first_chunk
ProbeRuntime.self_spec_draft_updown_first_margin_threshold = simulate_self_spec_gpu_pipeline_draft_updown_first_margin_threshold
at_exit { pipeline_router_trace_io.try(&.close) }
if simulate_mtp_self_draft_fusion > 0
  raise "MTP sidecar not found: #{mtp_path}" unless File.exists?(mtp_path)
  raise "--simulate-mtp-self-draft-fusion requires --simulate-logits-rank" if simulate_logit_rank.nil?
  raise "--simulate-mtp-self-draft-fusion requires --simulate-logits-layers" if simulate_logit_layers.empty?
  raise "--simulate-mtp-self-draft-fusion-topk must be positive" unless simulate_mtp_self_draft_fusion_topk > 0
  if rank = simulate_mtp_self_draft_fusion_updown_rank
    raise "--simulate-mtp-self-draft-fusion-updown must be positive" unless rank > 0
  end
end
if rank = simulate_self_draft_gpu_chain_updown_rank
  raise "--simulate-self-draft-gpu-chain-updown requires --simulate-self-draft-gpu-chain" unless simulate_self_draft_gpu_chain > 0
  raise "--simulate-self-draft-gpu-chain-updown requires --simulate-logits-rank" if simulate_logit_rank.nil?
  raise "--simulate-self-draft-gpu-chain-updown requires --simulate-logits-layers" if simulate_logit_layers.empty?
  raise "--simulate-self-draft-gpu-chain-updown must be positive" unless rank > 0
end
if simulate_self_spec_gpu_pipeline_mtp_k2_on_reject
  raise "MTP sidecar not found: #{mtp_path}" unless File.exists?(mtp_path)
  raise "--simulate-self-spec-gpu-pipeline-mtp-k2-on-reject requires --simulate-self-spec-gpu-pipeline or pipeline gammas/schedules" unless simulate_self_spec_gpu_pipeline > 0 || !simulate_self_spec_gpu_pipeline_gammas.empty? || !simulate_self_spec_gpu_pipeline_schedules.empty?
end
if !simulate_cost_truth_chunks.empty?
  raise "--simulate-cost-truth-table requires --simulate-logits-rank" if simulate_logit_rank.nil?
  raise "--simulate-cost-truth-table requires --simulate-logits-layers" if simulate_logit_layers.empty?
end
if simulate_lowrank_eval_suite
  raise "--simulate-lowrank-eval-suite requires --simulate-logits-rank" if simulate_logit_rank.nil?
  raise "--simulate-lowrank-eval-suite requires --simulate-logits-layers" if simulate_logit_layers.empty?
  raise "--simulate-lowrank-eval-suite self-spec gammas must be positive" if simulate_self_spec_gammas.any? { |v| v <= 0 }
end
if simulate_block_surrogate_start || simulate_block_surrogate_end
  raise "--simulate-block-residual-surrogate must set both start and end" unless simulate_block_surrogate_start && simulate_block_surrogate_end
end
raise "block surrogate clusters must be positive" unless simulate_block_surrogate_clusters > 0
raise "block surrogate delta basis list must not be empty" if simulate_block_surrogate_delta_basis_modes.empty?
valid_block_surrogate_delta_basis_modes = Set{"pca", "impact", "balanced"}
simulate_block_surrogate_delta_basis_modes.each do |mode|
  raise "unsupported block surrogate delta basis #{mode.inspect}; expected pca, impact, or balanced" unless valid_block_surrogate_delta_basis_modes.includes?(mode)
end
raise "block surrogate min ideal speedup must be positive" unless simulate_block_surrogate_min_ideal_speedup > 0.0
raise "block surrogate oracle generated calibration must be non-negative" unless simulate_block_surrogate_oracle_gen_calib >= 0
unless {"skip", "shadow"}.includes?(simulate_block_surrogate_state_mode)
  raise "--block-surrogate-state-mode must be skip or shadow"
end
unless simulate_block_surrogate_suite_blocks.empty?
  raise "--simulate-block-surrogate-suite-blocks requires --simulate-generate=N" unless simulate_generate_tokens > 0
  raise "--simulate-block-surrogate-suite-blocks requires --simulate-block-surrogate-self-spec-gammas=LIST" if simulate_block_surrogate_self_spec_gammas.empty?
end
unless simulate_block_surrogate_self_spec_gammas.empty?
  raise "--simulate-block-surrogate-self-spec-gammas requires --simulate-block-residual-surrogate or --simulate-block-surrogate-suite-blocks" unless (simulate_block_surrogate_start && simulate_block_surrogate_end) || !simulate_block_surrogate_suite_blocks.empty?
  raise "--simulate-block-surrogate-self-spec-gammas requires --simulate-generate=N" unless simulate_generate_tokens > 0
  raise "block surrogate self-spec gamma values must be positive" if simulate_block_surrogate_self_spec_gammas.any? { |v| v <= 0 }
end
if tree_k = simulate_block_surrogate_tree_oracle_k
  raise "--simulate-block-surrogate-tree-oracle requires K between 2 and 16" unless tree_k >= 2 && tree_k <= 16
  raise "--simulate-block-surrogate-tree-oracle requires --simulate-block-surrogate-self-spec-gammas=LIST as fixed schedules" if simulate_block_surrogate_self_spec_gammas.empty?
  raise "--simulate-block-surrogate-tree-oracle requires --simulate-generate=N" unless simulate_generate_tokens > 0
end
raise "--simulate-block-surrogate-tree-warmup must be non-negative" unless simulate_block_surrogate_tree_warmup_tokens >= 0
raise "--simulate-block-surrogate-tree-branch-verify and --simulate-block-surrogate-tree-select-advance are mutually exclusive" if simulate_block_surrogate_tree_branch_verify && simulate_block_surrogate_tree_select_advance
if topk_oracle_k = simulate_block_surrogate_topk_oracle_k
  raise "--simulate-block-surrogate-topk-oracle requires K between 2 and 16" unless topk_oracle_k >= 2 && topk_oracle_k <= 16
  raise "--simulate-block-surrogate-topk-oracle requires --simulate-generate=N>=4" unless simulate_generate_tokens >= 4
  raise "--simulate-block-surrogate-topk-oracle requires --simulate-block-residual-surrogate or --simulate-block-surrogate-suite-blocks" unless (simulate_block_surrogate_start && simulate_block_surrogate_end) || !simulate_block_surrogate_suite_blocks.empty?
end
if train_tokens = simulate_block_surrogate_topk_oracle_train_tokens
  raise "--simulate-block-surrogate-topk-oracle-train-tokens must be positive" unless train_tokens > 0
end

gguf = ML::GGUF::GGUFFile.new(model)
tok = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model, tokenizer_bin)
token_ids = token_ids_for_prompt(tok, prompt, tokens_limit, repeat: !prompt_as_prefix)
route_memory_store = nil.as(ML::GGUF::Qwen35PromptCache::Store?)
route_memory_entry = nil.as(ML::GGUF::Qwen35PromptCache::ProposalRouteEntry?)
route_memory_model_id = nil.as(String?)
route_memory_tokenizer_id = nil.as(String?)
route_memory_learned = false
if route_memory_root = simulate_self_spec_gpu_pipeline_draft_updown_route_memory_root
  resolution = ML::GGUF::Qwen35ProposalRoute.resolve(
    route_memory_root,
    model,
    tok,
    prompt,
    token_ids,
    simulate_self_spec_gpu_pipeline_draft_updown_route_key,
  )
  route_memory_model_id = resolution.model_id
  route_memory_tokenizer_id = resolution.tokenizer_id
  route_memory_store = resolution.store
  route_memory_entry = resolution.entry
  if route_hit = route_memory_entry
    case route_hit.route
    when ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_BASELINE
      simulate_self_spec_gpu_pipeline_draft_updown_rank = nil
      simulate_self_spec_gpu_pipeline_draft_updown_ranks.clear
      simulate_self_spec_gpu_pipeline_draft_updown_first_margin_threshold = nil
    when ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN
      simulate_self_spec_gpu_pipeline_draft_updown_rank = route_hit.route_rank
      simulate_self_spec_gpu_pipeline_draft_updown_ranks.clear
      simulate_self_spec_gpu_pipeline_draft_updown_first_margin_threshold = nil
      if simulate_self_spec_gpu_pipeline_draft_updown_layers.empty? && !route_hit.route_layers.empty?
        simulate_self_spec_gpu_pipeline_draft_updown_layers = route_hit.route_layers.dup
      end
    end
  end
end
ffn_pca_calib_token_sets = ffn_pca_calib_prompts.map { |calib_prompt| token_ids_for_prompt(tok, calib_prompt, tokens_limit, repeat: !prompt_as_prefix) }

weights = ML::GGUF::Qwen35Weights.from_gguf(model)
per_head = recurrent_k_vectors_for_prompt(weights, token_ids, layer_index)
samples = (simulate_delta || simulate_dn_regime_features || simulate_lowrank || simulate_lowrank_metal || simulate_lowrank_metal_project || simulate_lowrank_metal_chunk || simulate_lowrank_metal_chunk_out || simulate_lowrank_metal_layer_chunk || simulate_lowrank_metal_layer_full || simulate_lowrank_metal_layer_updown_rank || simulate_lowrank_metal_layer_overlap || simulate_lowrank_metal_verifier_overlap || simulate_lowrank_metal_decode_verifier_overlap || simulate_lowrank_metal_chunk_thread_overlap || simulate_multilayer_overlap_n > 0) ? recurrent_samples_for_prompt(weights, token_ids, layer_index) : [] of RecurrentSample
max_rank = ranks.max
if rank = simulate_logit_rank
  max_rank = Math.max(max_rank, rank)
end
calib_count = Math.min(calib_tokens, token_ids.size - 1)
raise "need at least one held-out token" unless calib_count > 0 && calib_count < token_ids.size

bases = per_head.map { |vectors| build_basis(vectors[0, calib_count], max_rank, basis_mode, pca_iters) }

puts "Qwen35 DeltaNet fixed-basis K residual probe"
puts "model=#{File.basename(model)}"
puts "layer=#{layer_index} token_vectors=#{token_ids.size} calib_tokens=#{calib_count} heldout_tokens=#{token_ids.size - calib_count}"
puts "heads=#{per_head.size} state_size=#{per_head[0][0].size} ranks=#{ranks.join(',')}"
puts "basis=#{basis_mode} pca_iters=#{pca_iters}; per-head basis over first calib_tokens; reports held-out L2 residual for normalized K vectors"
puts basis_rank_note(bases, max_rank)
if route_hit = route_memory_entry
  layers = route_hit.route_layers.empty? ? "default" : route_hit.route_layers.join(",")
  rank_text = route_hit.route_rank ? route_hit.route_rank.to_s : "na"
  key_text = simulate_self_spec_gpu_pipeline_draft_updown_route_key || "exact_prompt"
  puts "proposal_route_memory hit=1 key=#{key_text} route=#{route_hit.route} rank=#{rank_text} layers=#{layers} trigger=#{route_hit.trigger || "unknown"}"
elsif simulate_self_spec_gpu_pipeline_draft_updown_route_memory_root
  key_text = simulate_self_spec_gpu_pipeline_draft_updown_route_key || "exact_prompt"
  puts "proposal_route_memory hit=0 key=#{key_text}"
end
puts "thresholds=#{thresholds.map { |t| t.round(4) }.join(',')}"

if simulate_current_hidden_proposal
  proposal_token_sets = [{name: main_prompt_name, token_ids: token_ids}]
  simulate_current_hidden_proposal_suite_prompts.each do |suite_prompt|
    proposal_token_sets << {
      name:      suite_prompt[:name],
      token_ids: token_ids_for_prompt(tok, suite_prompt[:text], tokens_limit),
    }
  end
  proposal_token_sets.each do |item|
    ids = item[:token_ids]
    prompt_calib_count = Math.min(calib_tokens, ids.size - 1)
    raise "current-hidden proposal prompt #{item[:name]} needs at least one held-out token" unless prompt_calib_count > 0 && prompt_calib_count < ids.size
    proposal = run_current_hidden_proposal(weights, item[:name], ids, prompt_calib_count, simulate_current_hidden_proposal_topk, simulate_current_hidden_transition_rank, pca_iters)
    transition_per = proposal[:transition_samples] > 0 ? proposal[:transition_ms] / proposal[:transition_samples] : 0.0
    pca_transition_per = proposal[:pca_transition_samples] > 0 ? proposal[:pca_transition_ms] / proposal[:pca_transition_samples] : 0.0
    puts "current_hidden_proposal name=#{item[:name]} token_vectors=#{ids.size} calib_tokens=#{prompt_calib_count} heldout_tokens=#{ids.size - prompt_calib_count} top_k=#{proposal[:top_k]} train=#{proposal[:train_samples]} eval=#{proposal[:eval_samples]} unique_train_labels=#{proposal[:unique_train_labels]} top1=#{proposal[:top1_rate].round(2)}% topk=#{proposal[:topk_rate].round(2)}% hits=#{proposal[:top1_hits]}/#{proposal[:topk_hits]}/#{proposal[:eval_samples]} centroid_top1=#{proposal[:centroid_top1_rate].round(2)}% centroid_topk=#{proposal[:centroid_topk_rate].round(2)}% centroid_hits=#{proposal[:centroid_top1_hits]}/#{proposal[:centroid_topk_hits]}/#{proposal[:eval_samples]} collect_ms=#{proposal[:collect_ms].round(3)} proposal_ms=#{proposal[:proposal_ms].round(3)} proposal_ms_per_eval=#{(proposal[:proposal_ms] / proposal[:eval_samples]).round(6)} avg_best_cos=#{proposal[:avg_best_cos].round(6)} centroid_avg_best_cos=#{proposal[:centroid_avg_best_cos].round(6)} p50_best_cos=#{proposal[:p50_best_cos].round(6)} min_best_cos=#{proposal[:min_best_cos].round(6)} transition_samples=#{proposal[:transition_samples]} pca_transition_samples=#{proposal[:pca_transition_samples]} transition_label_top1=#{proposal[:transition_label_rate].round(2)}% transition_delta_top1=#{proposal[:transition_delta_rate].round(2)}% pca_transition_top1=#{proposal[:pca_transition_rate].round(2)}% transition_hits=#{proposal[:transition_label_hits]}/#{proposal[:transition_delta_hits]}/#{proposal[:transition_samples]} pca_transition_hits=#{proposal[:pca_transition_hits]}/#{proposal[:pca_transition_samples]} pca_transition_rank=#{proposal[:pca_transition_effective_rank]} transition_ms=#{proposal[:transition_ms].round(3)} pca_transition_ms=#{proposal[:pca_transition_ms].round(3)} transition_ms_per_eval=#{transition_per.round(6)} pca_transition_ms_per_eval=#{pca_transition_per.round(6)} note=proposal_only_exact_verifier_required"
    if simulate_generate_tokens > 0
      gen = run_current_hidden_generate_proposal(weights, item[:name], ids, simulate_generate_tokens, simulate_current_hidden_proposal_topk)
      puts "current_hidden_generate_proposal name=#{item[:name]} prompt_tokens=#{ids.size} gen_tokens=#{simulate_generate_tokens} top_k=#{gen[:top_k]} eval=#{gen[:eval_samples]} top1=#{gen[:top1_rate].round(2)}% topk=#{gen[:topk_rate].round(2)}% transition_label=#{gen[:transition_rate].round(2)}% hits=#{gen[:top1_hits]}/#{gen[:topk_hits]}/#{gen[:transition_hits]}/#{gen[:eval_samples]} collect_ms=#{gen[:collect_ms].round(3)} proposal_ms=#{gen[:proposal_ms].round(3)} proposal_ms_per_eval=#{(gen[:proposal_ms] / gen[:eval_samples]).round(6)} avg_best_cos=#{gen[:avg_best_cos].round(6)} exact_ids=#{gen[:exact_ids].join(',')} note=prompt_table_vs_exact_generated_hidden"
    end
  end
end

if simulate_dn_regime_features
  ranks.each do |rank|
    dn_regime_feature_notes("main", layer_index, rank, token_ids.size, calib_count, samples, bases, thresholds, dn_regime_g_cuts).each { |line| puts line }
  end
end

unless simulate_ffn_block_sparsity_layers.empty?
  sparse_layers = simulate_ffn_block_sparsity_layers.uniq.sort
  sparse_layers.each do |il|
    raise "--simulate-ffn-block-sparsity layer #{il} is out of range" unless il >= 0 && il < weights.layers.size
    raise "--simulate-ffn-block-sparsity layer #{il} is not recurrent" unless weights.layers[il].is_a?(ML::GGUF::Qwen35RecurrentWeights)
  end
  ffn_vectors = if ffn_pca_calib_token_sets.empty?
                  ffn_activation_vectors_for_prompt(weights, token_ids[0, calib_count], sparse_layers, calib_count)
                else
                  ffn_activation_vectors_for_token_sets(weights, ffn_pca_calib_token_sets, sparse_layers, calib_tokens)
                end
  sparsity_stats = [] of FFNBlockSparsityLayerStats
  sparse_layers.each do |il|
    vectors = ffn_vectors[il]? || [] of Array(Float64)
    raise "no FFN activation vectors captured for layer #{il}" if vectors.empty?
    sparsity_stats << ffn_block_sparsity_layer_stats(il, vectors, simulate_ffn_block_size)
  end
  print_ffn_block_sparsity_summary(sparsity_stats)
end

unless simulate_ffn_block_selector_layers.empty?
  selector_layers = simulate_ffn_block_selector_layers.uniq.sort
  selector_layers.each do |il|
    raise "--simulate-ffn-block-selector layer #{il} is out of range" unless il >= 0 && il < weights.layers.size
    raise "--simulate-ffn-block-selector layer #{il} is not recurrent" unless weights.layers[il].is_a?(ML::GGUF::Qwen35RecurrentWeights)
  end
  selector_samples = ffn_updown_samples_for_token_sets(weights, [token_ids], selector_layers, token_ids.size)
  selector_layers.each do |il|
    layer_samples = selector_samples[il]? || [] of NamedTuple(ffn_in: Array(Float64), activation: Array(Float64))
    train_count = Math.min(calib_count, layer_samples.size - 1)
    raise "no held-out FFN selector samples captured for layer #{il}" unless train_count > 0 && train_count < layer_samples.size
    print_ffn_block_selector_stats(il, layer_samples, train_count, simulate_ffn_block_size, simulate_ffn_block_selector_percents)
  end
end

unless simulate_block_surrogate_suite_blocks.empty?
  block_rank = simulate_block_surrogate_rank || simulate_logit_rank || ranks.max
  raise "block surrogate rank must be positive" unless block_rank > 0
  suite_token_sets = [{name: main_prompt_name, token_ids: token_ids}] of PromptTokenSet
  simulate_block_surrogate_suite_prompts.each do |suite_prompt|
    suite_token_sets << {
      name:      suite_prompt[:name],
      token_ids: token_ids_for_prompt(tok, suite_prompt[:text], tokens_limit),
    }
  end
  run_block_surrogate_suite(weights, suite_token_sets, simulate_block_surrogate_suite_blocks,
    block_rank, pca_iters, calib_tokens, simulate_generate_tokens,
    simulate_block_surrogate_self_spec_gammas, simulate_block_surrogate_clusters,
    simulate_block_surrogate_delta_basis_modes, simulate_block_surrogate_state_mode,
    simulate_block_surrogate_oracle_gen_calib,
    simulate_block_surrogate_tree_oracle_k, simulate_block_surrogate_tree_warmup_tokens,
    simulate_block_surrogate_tree_prefill_seed, simulate_block_surrogate_tree_branch_verify,
    simulate_block_surrogate_tree_select_advance, simulate_block_surrogate_topk_oracle_k,
    simulate_block_surrogate_topk_oracle_train_tokens, simulate_block_surrogate_min_ideal_speedup)
end

if block_start = simulate_block_surrogate_start
  block_end = simulate_block_surrogate_end.not_nil!
  block_rank = simulate_block_surrogate_rank || simulate_logit_rank || ranks.max
  raise "block surrogate rank must be positive" unless block_rank > 0
  raise "block surrogate end must be within layer count" unless block_end < weights.layers.size
  t0 = Time.instant
  block_samples = collect_block_residual_samples(weights, token_ids, block_start, block_end)
  collect_ms = (Time.instant - t0).total_milliseconds
  train_samples = block_samples[0, calib_count]
  block_delta_basis_mode = simulate_block_surrogate_delta_basis_modes[0]
  block_impact_basis = block_delta_basis_mode == "pca" ? [] of Array(Float64) : output_margin_impact_vectors(weights, token_ids[0, calib_count])
  t_train = Time.instant
  block_adapter = train_block_residual_surrogate(train_samples, block_start, block_end, block_rank, pca_iters,
    delta_basis_mode: block_delta_basis_mode, impact_basis_seed: block_impact_basis)
  train_ms = (Time.instant - t_train).total_milliseconds
  stats = block_residual_surrogate_stats(block_samples, block_adapter, calib_count)
  puts "block_residual_surrogate_static block=#{block_start}:#{block_end} delta_basis=#{block_delta_basis_mode} impact_vectors=#{block_impact_basis.size} rank=#{block_rank} effective_input_rank=#{block_adapter.input_basis.size} effective_delta_rank=#{block_adapter.delta_basis.size} calib=#{calib_count} heldout=#{stats[:count]} hidden_cos_mean=#{stats[:mean_cos].round(8)} hidden_cos_min=#{stats[:min_cos].round(8)} delta_cos_mean=#{stats[:mean_delta_cos].round(8)} rmse=#{stats[:rmse].round(8)} rel_rmse=#{stats[:rel_rmse].round(8)} delta_rel_rmse=#{stats[:delta_rel_rmse].round(8)} residual_energy=#{stats[:residual_energy].round(8)} max_delta=#{stats[:max_delta].round(6)} adapter_ms=#{stats[:adapter_ms].round(3)} adapter_ms_per_sample=#{stats[:adapter_ms_per_sample].round(6)} collect_ms=#{collect_ms.round(3)} train_ms=#{train_ms.round(3)} note=teacher_forced_exact_trajectory_not_state_replacement"
  simulate_block_surrogate_error_feedback_decays.each do |decay|
    fb = block_residual_error_feedback_stats(block_samples, block_adapter, calib_count, decay)
    rel_gain = stats[:rel_rmse] > 0.0 ? 100.0 * (stats[:rel_rmse] - fb[:rel_rmse]) / stats[:rel_rmse] : 0.0
    delta_gain = stats[:delta_rel_rmse] > 0.0 ? 100.0 * (stats[:delta_rel_rmse] - fb[:delta_rel_rmse]) / stats[:delta_rel_rmse] : 0.0
    puts "block_residual_error_feedback block=#{block_start}:#{block_end} mode=global rank=#{block_rank} decay=#{decay} calib_warmup=#{calib_count} heldout=#{fb[:count]} hidden_cos_mean=#{fb[:mean_cos].round(8)} hidden_cos_min=#{fb[:min_cos].round(8)} delta_cos_mean=#{fb[:mean_delta_cos].round(8)} rmse=#{fb[:rmse].round(8)} rel_rmse=#{fb[:rel_rmse].round(8)} delta_rel_rmse=#{fb[:delta_rel_rmse].round(8)} max_delta=#{fb[:max_delta].round(6)} adapter_ms=#{fb[:adapter_ms].round(3)} adapter_ms_per_sample=#{fb[:adapter_ms_per_sample].round(6)} rel_rmse_gain_pct=#{rel_gain.round(2)} delta_rel_rmse_gain_pct=#{delta_gain.round(2)} note=one_token_lag_adaptive_filter_exact_observations"
  end
  if simulate_block_surrogate_clusters > 1
    t_mix = Time.instant
    mixture = train_block_residual_mixture(train_samples, block_start, block_end, block_rank, simulate_block_surrogate_clusters, pca_iters)
    mix_train_ms = (Time.instant - t_mix).total_milliseconds
    mix_stats = block_residual_mixture_stats(block_samples, mixture, calib_count)
    puts "block_residual_surrogate_mixture block=#{block_start}:#{block_end} rank=#{block_rank} clusters=#{mixture.centroids.size} requested_clusters=#{simulate_block_surrogate_clusters} cluster_sizes=#{mixture.cluster_sizes.join(',')} calib=#{calib_count} heldout=#{mix_stats[:count]} hidden_cos_mean=#{mix_stats[:mean_cos].round(8)} hidden_cos_min=#{mix_stats[:min_cos].round(8)} delta_cos_mean=#{mix_stats[:mean_delta_cos].round(8)} rmse=#{mix_stats[:rmse].round(8)} rel_rmse=#{mix_stats[:rel_rmse].round(8)} delta_rel_rmse=#{mix_stats[:delta_rel_rmse].round(8)} residual_energy=#{mix_stats[:residual_energy].round(8)} max_delta=#{mix_stats[:max_delta].round(6)} train_ms=#{mix_train_ms.round(3)} note=nearest_input_pca_centroid_teacher_forced_static"
    simulate_block_surrogate_error_feedback_decays.each do |decay|
      fb = block_residual_error_feedback_stats(block_samples, mixture, calib_count, decay)
      rel_gain = mix_stats[:rel_rmse] > 0.0 ? 100.0 * (mix_stats[:rel_rmse] - fb[:rel_rmse]) / mix_stats[:rel_rmse] : 0.0
      delta_gain = mix_stats[:delta_rel_rmse] > 0.0 ? 100.0 * (mix_stats[:delta_rel_rmse] - fb[:delta_rel_rmse]) / mix_stats[:delta_rel_rmse] : 0.0
      puts "block_residual_error_feedback block=#{block_start}:#{block_end} mode=mixture rank=#{block_rank} clusters=#{mixture.centroids.size} decay=#{decay} calib_warmup=#{calib_count} heldout=#{fb[:count]} hidden_cos_mean=#{fb[:mean_cos].round(8)} hidden_cos_min=#{fb[:min_cos].round(8)} delta_cos_mean=#{fb[:mean_delta_cos].round(8)} rmse=#{fb[:rmse].round(8)} rel_rmse=#{fb[:rel_rmse].round(8)} delta_rel_rmse=#{fb[:delta_rel_rmse].round(8)} max_delta=#{fb[:max_delta].round(6)} adapter_ms=#{fb[:adapter_ms].round(3)} adapter_ms_per_sample=#{fb[:adapter_ms_per_sample].round(6)} rel_rmse_gain_pct=#{rel_gain.round(2)} delta_rel_rmse_gain_pct=#{delta_gain.round(2)} note=one_token_lag_adaptive_filter_exact_observations"
    end
    if simulate_block_surrogate_policy
      mix_logit = simulate_block_surrogate_logits_policy(weights, token_ids, block_start, block_end, mixture, calib_count, simulate_block_surrogate_state_mode)
      puts "block_surrogate_logit_policy block=#{block_start}:#{block_end} mode=mixture state_mode=#{simulate_block_surrogate_state_mode} rank=#{block_rank} clusters=#{mixture.centroids.size} top1_match=#{mix_logit[:top1_match].round(2)}% top5_hit=#{mix_logit[:top5_hit].round(2)}% mean_cos=#{mix_logit[:mean_cos].round(8)} min_cos=#{mix_logit[:min_cos].round(8)} mean_kl=#{mix_logit[:mean_kl].round(8)} max_kl=#{mix_logit[:max_kl].round(8)} min_margin=#{mix_logit[:min_margin].round(6)} confident_mismatches=#{mix_logit[:confident_mismatches]} approx_blocks=#{mix_logit[:approx_blocks]} skipped_layers=#{mix_logit[:skipped_layers]}"
      if simulate_generate_tokens > 0
        mix_gen = simulate_block_surrogate_greedy_policy(weights, token_ids, simulate_generate_tokens, block_start, block_end, mixture, calib_count, simulate_block_surrogate_state_mode)
        puts "block_surrogate_greedy_policy block=#{block_start}:#{block_end} mode=mixture state_mode=#{simulate_block_surrogate_state_mode} rank=#{block_rank} clusters=#{mixture.centroids.size} gen_tokens=#{simulate_generate_tokens} top1_match=#{mix_gen[:top1_match].round(2)}% top5_hit=#{mix_gen[:top5_hit].round(2)}% mean_cos=#{mix_gen[:mean_cos].round(8)} min_cos=#{mix_gen[:min_cos].round(8)} mean_kl=#{mix_gen[:mean_kl].round(8)} max_kl=#{mix_gen[:max_kl].round(8)} min_margin=#{mix_gen[:min_margin].round(6)} confident_mismatches=#{mix_gen[:confident_mismatches]} approx_blocks=#{mix_gen[:approx_blocks]} skipped_layers=#{mix_gen[:skipped_layers]} exact_ids=#{mix_gen[:exact_ids].join(',')} approx_ids=#{mix_gen[:approx_ids].join(',')}"
      end
    end
    simulate_block_surrogate_self_spec_gammas.each do |gamma|
      spec = simulate_block_surrogate_self_spec_policy(weights, token_ids, simulate_generate_tokens, gamma, block_start, block_end, mixture, calib_count, simulate_block_surrogate_state_mode)
      puts "block_surrogate_self_spec block=#{block_start}:#{block_end} mode=mixture state_mode=#{simulate_block_surrogate_state_mode} rank=#{block_rank} clusters=#{mixture.centroids.size} gamma=#{gamma} gen_tokens=#{simulate_generate_tokens} chunks=#{spec[:chunks]} full_accept_chunks=#{spec[:full_accept_chunks]} rejections=#{spec[:rejections]} accepted_draft_tokens=#{spec[:accepted_draft_tokens]} proposed_tokens=#{spec[:proposed_tokens]} accept_rate=#{spec[:accept_rate].round(2)}% avg_accept=#{spec[:avg_accept].round(3)} verifier_tokens=#{spec[:verifier_tokens]} correction_steps=#{spec[:correction_steps]} draft_top2_hit=#{spec[:draft_top2_hit_rate].round(2)}% draft_top5_hit=#{spec[:draft_top5_hit_rate].round(2)}% baseline_decode_ms=#{spec[:baseline_decode_ms].round(3)} draft_ms=#{spec[:draft_ms].round(3)} verifier_ms=#{spec[:verifier_ms].round(3)} self_seq_decode_ms=#{spec[:self_seq_decode_ms].round(3)} ideal_overlap_decode_ms=#{spec[:ideal_overlap_decode_ms].round(3)} cpu_seq_speedup=#{spec[:cpu_seq_speedup].round(4)} ideal_overlap_speedup=#{spec[:ideal_overlap_speedup].round(4)} parity=#{spec[:parity]} verifier_parity=#{spec[:verifier_parity]} gamma_history=#{spec[:gamma_history].join(',')} accept_history=#{spec[:accept_history].join(',')} draft_min_margin_history=#{spec[:draft_min_margin_history].map { |v| v.round(4) }.join(',')} exact_ids=#{spec[:exact_ids].join(',')} emitted_ids=#{spec[:emitted_ids].join(',')} baseline_ids=#{spec[:baseline_ids].join(',')} draft_ids=#{spec[:draft_ids].join(',')}"
      if top_k = simulate_block_surrogate_tree_oracle_k
      tree = simulate_block_surrogate_tree_oracle(weights, token_ids, simulate_generate_tokens, top_k, [gamma],
        block_start, block_end, mixture, calib_count, simulate_block_surrogate_state_mode,
        simulate_block_surrogate_tree_warmup_tokens, simulate_block_surrogate_tree_prefill_seed,
        simulate_block_surrogate_tree_branch_verify, simulate_block_surrogate_tree_select_advance)
        parity = tree[:exact_ids] == tree[:emitted_ids]
        puts "block_surrogate_tree_oracle block=#{block_start}:#{block_end} mode=mixture state_mode=#{simulate_block_surrogate_state_mode} rank=#{block_rank} clusters=#{mixture.centroids.size} top_k=#{top_k} gamma=#{gamma} gen_tokens=#{simulate_generate_tokens} parity=#{parity} #{block_surrogate_tree_metrics_summary(tree)}"
      end
    end
    if top_k = simulate_block_surrogate_topk_oracle_k
      oracle = simulate_block_surrogate_topk_oracle_calibration(weights, token_ids, simulate_generate_tokens, top_k,
        simulate_block_surrogate_topk_oracle_train_tokens, block_start, block_end, mixture, calib_count,
        simulate_block_surrogate_state_mode)
      print_block_surrogate_topk_oracle("block_surrogate_topk_oracle block=#{block_start}:#{block_end} mode=mixture state_mode=#{simulate_block_surrogate_state_mode} rank=#{block_rank} clusters=#{mixture.centroids.size}",
        oracle, top_k, simulate_generate_tokens)
    end
  end
  if simulate_block_surrogate_policy
    logit = simulate_block_surrogate_logits_policy(weights, token_ids, block_start, block_end, block_adapter, calib_count, simulate_block_surrogate_state_mode)
    puts "block_surrogate_logit_policy block=#{block_start}:#{block_end} mode=global state_mode=#{simulate_block_surrogate_state_mode} rank=#{block_rank} clusters=1 top1_match=#{logit[:top1_match].round(2)}% top5_hit=#{logit[:top5_hit].round(2)}% mean_cos=#{logit[:mean_cos].round(8)} min_cos=#{logit[:min_cos].round(8)} mean_kl=#{logit[:mean_kl].round(8)} max_kl=#{logit[:max_kl].round(8)} min_margin=#{logit[:min_margin].round(6)} confident_mismatches=#{logit[:confident_mismatches]} approx_blocks=#{logit[:approx_blocks]} skipped_layers=#{logit[:skipped_layers]}"
    if simulate_generate_tokens > 0
      gen = simulate_block_surrogate_greedy_policy(weights, token_ids, simulate_generate_tokens, block_start, block_end, block_adapter, calib_count, simulate_block_surrogate_state_mode)
      puts "block_surrogate_greedy_policy block=#{block_start}:#{block_end} mode=global state_mode=#{simulate_block_surrogate_state_mode} rank=#{block_rank} clusters=1 gen_tokens=#{simulate_generate_tokens} top1_match=#{gen[:top1_match].round(2)}% top5_hit=#{gen[:top5_hit].round(2)}% mean_cos=#{gen[:mean_cos].round(8)} min_cos=#{gen[:min_cos].round(8)} mean_kl=#{gen[:mean_kl].round(8)} max_kl=#{gen[:max_kl].round(8)} min_margin=#{gen[:min_margin].round(6)} confident_mismatches=#{gen[:confident_mismatches]} approx_blocks=#{gen[:approx_blocks]} skipped_layers=#{gen[:skipped_layers]} exact_ids=#{gen[:exact_ids].join(',')} approx_ids=#{gen[:approx_ids].join(',')}"
    end
  end
  simulate_block_surrogate_self_spec_gammas.each do |gamma|
    spec = simulate_block_surrogate_self_spec_policy(weights, token_ids, simulate_generate_tokens, gamma, block_start, block_end, block_adapter, calib_count, simulate_block_surrogate_state_mode)
    puts "block_surrogate_self_spec block=#{block_start}:#{block_end} mode=global state_mode=#{simulate_block_surrogate_state_mode} rank=#{block_rank} clusters=1 gamma=#{gamma} gen_tokens=#{simulate_generate_tokens} chunks=#{spec[:chunks]} full_accept_chunks=#{spec[:full_accept_chunks]} rejections=#{spec[:rejections]} accepted_draft_tokens=#{spec[:accepted_draft_tokens]} proposed_tokens=#{spec[:proposed_tokens]} accept_rate=#{spec[:accept_rate].round(2)}% avg_accept=#{spec[:avg_accept].round(3)} verifier_tokens=#{spec[:verifier_tokens]} correction_steps=#{spec[:correction_steps]} draft_top2_hit=#{spec[:draft_top2_hit_rate].round(2)}% draft_top5_hit=#{spec[:draft_top5_hit_rate].round(2)}% baseline_decode_ms=#{spec[:baseline_decode_ms].round(3)} draft_ms=#{spec[:draft_ms].round(3)} verifier_ms=#{spec[:verifier_ms].round(3)} self_seq_decode_ms=#{spec[:self_seq_decode_ms].round(3)} ideal_overlap_decode_ms=#{spec[:ideal_overlap_decode_ms].round(3)} cpu_seq_speedup=#{spec[:cpu_seq_speedup].round(4)} ideal_overlap_speedup=#{spec[:ideal_overlap_speedup].round(4)} parity=#{spec[:parity]} verifier_parity=#{spec[:verifier_parity]} gamma_history=#{spec[:gamma_history].join(',')} accept_history=#{spec[:accept_history].join(',')} draft_min_margin_history=#{spec[:draft_min_margin_history].map { |v| v.round(4) }.join(',')} exact_ids=#{spec[:exact_ids].join(',')} emitted_ids=#{spec[:emitted_ids].join(',')} baseline_ids=#{spec[:baseline_ids].join(',')} draft_ids=#{spec[:draft_ids].join(',')}"
    if top_k = simulate_block_surrogate_tree_oracle_k
      tree = simulate_block_surrogate_tree_oracle(weights, token_ids, simulate_generate_tokens, top_k, [gamma],
        block_start, block_end, block_adapter, calib_count, simulate_block_surrogate_state_mode,
        simulate_block_surrogate_tree_warmup_tokens, simulate_block_surrogate_tree_prefill_seed,
        simulate_block_surrogate_tree_branch_verify, simulate_block_surrogate_tree_select_advance)
      parity = tree[:exact_ids] == tree[:emitted_ids]
      puts "block_surrogate_tree_oracle block=#{block_start}:#{block_end} mode=global state_mode=#{simulate_block_surrogate_state_mode} rank=#{block_rank} clusters=1 top_k=#{top_k} gamma=#{gamma} gen_tokens=#{simulate_generate_tokens} parity=#{parity} #{block_surrogate_tree_metrics_summary(tree)}"
    end
  end
  if top_k = simulate_block_surrogate_topk_oracle_k
    oracle = simulate_block_surrogate_topk_oracle_calibration(weights, token_ids, simulate_generate_tokens, top_k,
      simulate_block_surrogate_topk_oracle_train_tokens, block_start, block_end, block_adapter, calib_count,
      simulate_block_surrogate_state_mode)
    print_block_surrogate_topk_oracle("block_surrogate_topk_oracle block=#{block_start}:#{block_end} mode=global state_mode=#{simulate_block_surrogate_state_mode} rank=#{block_rank} clusters=1",
      oracle, top_k, simulate_generate_tokens)
  end
end

if rank = simulate_logit_rank
  if simulate_logit_layers.empty?
    logit = simulate_logits(weights, token_ids, layer_index, bases, rank, calib_count)
    puts "logit_drift rank=#{rank} mean_cos=#{logit[:mean_cos].round(8)} min_cos=#{logit[:min_cos].round(8)} max_delta=#{logit[:max_delta].round(6)} top1_match=#{logit[:top1_match].round(2)}%"
  else
    sorted_simulate_logit_layers = simulate_logit_layers.uniq.sort
    layer_vectors = {} of Int32 => BasisSet
    layer_bases = {} of Int32 => BasisSet
    sorted_simulate_logit_layers.each do |il|
      vectors = il == layer_index ? per_head : recurrent_k_vectors_for_prompt(weights, token_ids, il)
      layer_vectors[il] = vectors
      layer_bases[il] = if il == layer_index
                          bases
                        else
                          vectors.map do |head_vectors|
                            build_basis(head_vectors[0, calib_count], max_rank, basis_mode, pca_iters)
                          end
                        end
    end
    rank_notes = sorted_simulate_logit_layers.map do |il|
      "#{il}:#{basis_rank_note(layer_bases[il], rank)}"
    end
    puts "layer_basis_effective_ranks #{rank_notes.join(' ')}"
    if simulate_self_spec_gpu_pipeline_route_features
      puts prompt_route_feature_note("main", sorted_simulate_logit_layers, rank, token_ids.size, calib_count, layer_vectors, layer_bases, thresholds)
      prompt_route_layer_feature_notes("main", sorted_simulate_logit_layers, rank, token_ids.size, calib_count, layer_vectors, layer_bases, thresholds).each { |line| puts line }
    end
    ffn_pca_ranks = [] of Int32
    ffn_pca_down_ranks = [] of Int32
    ffn_pca_updown_ranks = [] of Int32
    ffn_block_pred_percents = [] of Int32
    if metal_updown_rank = simulate_ffn_updown_metal_rank
      ffn_pca_updown_ranks << metal_updown_rank
    end
    if cost_updown_rank = simulate_cost_truth_updown_rank
      ffn_pca_updown_ranks << cost_updown_rank
    end
    if layer_updown_rank = simulate_lowrank_metal_layer_updown_rank
      ffn_pca_updown_ranks << layer_updown_rank
    end
    if pipeline_updown_rank = simulate_self_spec_gpu_pipeline_draft_updown_rank
      ffn_pca_updown_ranks << pipeline_updown_rank
    end
    if chain_updown_rank = simulate_self_draft_gpu_chain_updown_rank
      ffn_pca_updown_ranks << chain_updown_rank
    end
    if fusion_updown_rank = simulate_mtp_self_draft_fusion_updown_rank
      ffn_pca_updown_ranks << fusion_updown_rank
    end
    simulate_self_spec_gpu_pipeline_draft_updown_ranks.each do |pipeline_updown_rank|
      ffn_pca_updown_ranks << pipeline_updown_rank if pipeline_updown_rank > 0
    end
    simulate_cheap_self_draft_variants.each do |variant|
      if pca_rank = draft_variant_ffn_pca_rank(variant)
        ffn_pca_ranks << pca_rank
      end
      if pca_down_rank = draft_variant_ffn_pca_down_rank(variant)
        ffn_pca_down_ranks << pca_down_rank
      end
      if pca_updown_rank = draft_variant_ffn_pca_updown_rank(variant)
        ffn_pca_updown_ranks << pca_updown_rank
      end
      if block_pred_percent = draft_variant_ffn_block_pred_percent(variant)
        ffn_block_pred_percents << block_pred_percent
      end
    end
    ffn_activation_bases = nil.as(FFNBasisMap?)
    ffn_down_adapters = nil.as(FFNAdapterMap?)
    ffn_updown_adapters = nil.as(FFNUpDownAdapterMap?)
    ffn_block_selectors = nil.as(FFNBlockSelectorMap?)
    unless ffn_block_pred_percents.empty?
      selector_token_sets = ffn_pca_calib_token_sets.empty? ? [token_ids[0, calib_count]] : ffn_pca_calib_token_sets
      selector_token_count = ffn_pca_calib_token_sets.empty? ? calib_count : calib_tokens
      selector_samples = ffn_updown_samples_for_token_sets(weights, selector_token_sets, sorted_simulate_logit_layers, selector_token_count)
      selectors = {} of Int32 => FFNBlockSelector
      sorted_simulate_logit_layers.each do |il|
        samples_for_layer = selector_samples[il]? || [] of FFNActivationSample
        raise "no FFN block selector samples captured for layer #{il}" if samples_for_layer.empty?
        selectors[il] = train_ffn_block_selector(samples_for_layer, ffn_block_pred_percents, simulate_ffn_block_size)
      end
      ffn_block_selectors = selectors
      calib_source = ffn_pca_calib_token_sets.empty? ? "eval_prompt_prefix" : "external_prompts:#{ffn_pca_calib_token_sets.size}"
      puts "ffn_block_selector_adapter source=#{calib_source} layers=#{selectors.keys.sort.join(',')} percents=#{ffn_block_pred_percents.uniq.sort.join(',')} block_size=#{simulate_ffn_block_size} samples=#{selector_samples.map { |il, s| "#{il}:#{s.size}" }.join(',')}"
    end
    all_ffn_pca_ranks = ffn_pca_ranks + ffn_pca_down_ranks + ffn_pca_updown_ranks
    unless all_ffn_pca_ranks.empty?
      max_ffn_pca_rank = all_ffn_pca_ranks.max
      ffn_vectors = if ffn_pca_calib_token_sets.empty?
                      ffn_activation_vectors_for_prompt(weights, token_ids, simulate_logit_layers.uniq, calib_count)
                    else
                      ffn_activation_vectors_for_token_sets(weights, ffn_pca_calib_token_sets, simulate_logit_layers.uniq, calib_tokens)
                    end
      built = {} of Int32 => Array(Array(Float64))
      ffn_vectors.each do |il, vectors|
        next if vectors.empty?
        built[il] = pca_basis(vectors, max_ffn_pca_rank, pca_iters)
      end
      ffn_activation_bases = built
      calib_source = ffn_pca_calib_token_sets.empty? ? "eval_prompt" : "external_prompts:#{ffn_pca_calib_token_sets.size}"
      puts "ffn_activation_pca_basis source=#{calib_source} layers=#{built.keys.sort.join(',')} max_rank=#{max_ffn_pca_rank} calib_vectors=#{built.map { |il, _| "#{il}:#{ffn_vectors[il].size}" }.join(',')} pca_iters=#{pca_iters}"
      unless (ffn_pca_down_ranks + ffn_pca_updown_ranks).empty?
        adapters = {} of Int32 => FFNAdapter
        built.each do |il, basis_set|
          layer = weights.layers[il].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "FFN PCA-down layer #{il} is not recurrent"
          down_basis = basis_set.map do |basis_vec|
            ML::GGUF::Qwen35CPU.qmatvec_nobias(layer.ffn_down_qw, basis_vec.map(&.to_f32))
          end
          adapters[il] = FFNAdapter.new(basis_set, down_basis)
        end
        ffn_down_adapters = adapters
        adapter_rank_note = (ffn_pca_down_ranks + ffn_pca_updown_ranks).max
        puts "ffn_down_pca_adapter layers=#{adapters.keys.sort.join(',')} max_rank=#{adapter_rank_note} precomputed_vectors=#{adapters.map { |il, adapter| "#{il}:#{adapter.down_basis.size}" }.join(',')}"
      end
      unless ffn_pca_updown_ranks.empty?
        updown_token_sets = ffn_pca_calib_token_sets.empty? ? [token_ids[0, calib_count]] : ffn_pca_calib_token_sets
        updown_samples = ffn_updown_samples_for_token_sets(weights, updown_token_sets, simulate_logit_layers.uniq, calib_tokens)
        updown = {} of Int32 => FFNUpDownAdapter
        max_updown_rank = ffn_pca_updown_ranks.max
        down_adapters = ffn_down_adapters || raise "FFN up/down adapter requires down adapters"
        built.each do |il, basis_set|
          samples_for_layer = updown_samples[il]? || [] of NamedTuple(ffn_in: Array(Float64), activation: Array(Float64))
          updown[il] = train_ffn_updown_adapter(samples_for_layer, basis_set, down_adapters[il].down_basis, max_updown_rank)
        end
        if quant_bits = ffn_updown_adapter_quant_bits
          quant_block = ffn_updown_adapter_quant_hadamard_block
          updown = updown.transform_values do |adapter|
            quantized_updown_adapter(adapter, quant_bits, quant_block)
          end
          mode = quant_block ? "hadamard#{quant_block}_q#{quant_bits}" : "raw_q#{quant_bits}"
          puts "ffn_updown_pca_adapter_quant mode=#{mode} layers=#{updown.keys.sort.join(',')} max_rank=#{max_updown_rank}"
        end
        ffn_updown_adapters = updown
        puts "ffn_updown_pca_adapter layers=#{updown.keys.sort.join(',')} max_rank=#{max_updown_rank} samples=#{updown_samples.map { |il, s| "#{il}:#{s.size}" }.join(',')}"
        if dump_path = dump_ffn_updown_adapters_path
          dump_ffn_updown_adapters(dump_path.not_nil!, updown, max_updown_rank, weights.hparams.n_embd, ffn_pca_calib_token_sets.empty? ? "eval_prompt_prefix" : "external_prompts:#{ffn_pca_calib_token_sets.size}")
          puts "ffn_updown_pca_adapter_dump path=#{dump_path} layers=#{updown.keys.sort.join(',')} rank=#{max_updown_rank} hidden=#{weights.hparams.n_embd}"
        end
        if simulate_self_spec_gpu_pipeline_ffn_updown_route_features
          puts ffn_updown_route_feature_note("main", weights, token_ids, calib_count, sorted_simulate_logit_layers, updown, max_updown_rank)
        end
        if simulate_self_spec_gpu_pipeline_ffn_updown_hadamard_quant_features
          ffn_updown_hadamard_quant_feature_note("main", weights, token_ids, calib_count, sorted_simulate_logit_layers, updown, max_updown_rank,
            simulate_self_spec_gpu_pipeline_ffn_updown_hadamard_bits,
            simulate_self_spec_gpu_pipeline_ffn_updown_hadamard_blocks).each { |line| puts line }
        end
        if metal_rank = simulate_ffn_updown_metal_rank
          raise "Metal FFN up/down unavailable" unless ML::GGUF::Qwen35Metal.available?
          layer_id = simulate_logit_layers.uniq.find { |il| updown[il]? && updown_samples[il]? && !updown_samples[il].empty? } || raise "no FFN up/down sample for Metal gate"
          adapter = updown[layer_id]
          sample = updown_samples[layer_id][0]
          ffn_in = sample[:ffn_in].map(&.to_f32)
          hidden_dim = ffn_in.size
          bench_rank = Math.min(metal_rank, adapter.coeff_weights.size)
          raise "FFN up/down Metal rank must be positive" unless bench_rank > 0
          raise "FFN up/down Metal output dim mismatch" unless adapter.down_basis[0].size == hidden_dim

          x_mean = adapter.x_mean.map(&.to_f32)
          c_mean = adapter.c_mean.map(&.to_f32)
          coeff_weights = Array(Float32).new(bench_rank * hidden_dim)
          down_basis = Array(Float32).new(bench_rank * hidden_dim)
          bench_rank.times do |j|
            hidden_dim.times { |d| coeff_weights << adapter.coeff_weights[j][d].to_f32 }
            hidden_dim.times { |d| down_basis << adapter.down_basis[j][d] }
          end

          cpu_out = [] of Float32
          cpu_reps = 3
          t_cpu = Time.instant
          cpu_reps.times { cpu_out = ffn_out_from_updown_adapter(ffn_in, adapter, bench_rank) }
          cpu_ms = (Time.instant - t_cpu).total_milliseconds / cpu_reps

          x_mean_buf = ML::MetalBuffer.from_array(x_mean[0, hidden_dim])
          c_mean_buf = ML::MetalBuffer.from_array(c_mean[0, bench_rank])
          coeff_w_buf = ML::MetalBuffer.from_array(coeff_weights)
          down_buf = ML::MetalBuffer.from_array(down_basis)

          metal_out = ML::GGUF::Qwen35Metal.ffn_pca_updown_out_resident(ffn_in, x_mean_buf, c_mean_buf, coeff_w_buf, down_buf, hidden_dim, bench_rank)
          metal_reps = 5
          t_metal = Time.instant
          metal_reps.times { metal_out = ML::GGUF::Qwen35Metal.ffn_pca_updown_out_resident(ffn_in, x_mean_buf, c_mean_buf, coeff_w_buf, down_buf, hidden_dim, bench_rank) }
          metal_ms = (Time.instant - t_metal).total_milliseconds / metal_reps

          sum_sq = 0.0
          max_delta = 0.0
          hidden_dim.times do |d|
            delta = (cpu_out[d] - metal_out[d]).abs.to_f64
            max_delta = delta if delta > max_delta
            sum_sq += delta * delta
          end
          rmse = Math.sqrt(sum_sq / hidden_dim)
          puts "ffn_updown_metal layer=#{layer_id} rank=#{bench_rank} hidden=#{hidden_dim} max_delta=#{max_delta.round(8)} rmse=#{rmse.round(8)} cpu_ms=#{cpu_ms.round(4)} metal_ms=#{metal_ms.round(4)} metal_note=resident_adapter_upload_x_readback"
        end
      end
    end
    unless simulate_cost_truth_chunks.empty?
      cost_updown_layers = simulate_cost_truth_updown_layers.empty? ? nil : Set(Int32).new(simulate_cost_truth_updown_layers)
      simulate_self_spec_cost_truth_table(weights, token_ids, calib_count, simulate_cost_truth_chunks, layer_bases, rank,
        ffn_updown_adapters, simulate_cost_truth_updown_rank, cost_updown_layers, simulate_cost_truth_branch_split_guards)
    end
    thresholds_to_run = if simulate_fallback_thresholds.empty?
                          [simulate_fallback_threshold]
                        else
                          simulate_fallback_thresholds.map { |v| v.as(Float64?) }
                        end
    thresholds_to_run.each do |fallback_threshold|
      logit = simulate_logits_policy(weights, token_ids, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, simulate_oracle_refresh_interval, simulate_output_margin_threshold)
      total_steps = logit[:approx_steps] + logit[:fallback_steps]
      approx_rate = total_steps > 0 ? (100.0 * logit[:approx_steps] / total_steps) : 0.0
      fallback_score_note = ProbeRuntime.fallback_score_mode == "raw" ? "" : " fallback_score=#{ProbeRuntime.fallback_score_mode}"
      fallback_note = fallback_threshold ? " fallback_threshold=#{fallback_threshold}#{fallback_score_note} approx_rate=#{approx_rate.round(2)}%" : fallback_score_note
      output_note = simulate_output_margin_threshold ? " output_margin_threshold=#{simulate_output_margin_threshold} output_fallbacks=#{logit[:output_fallbacks]}" : ""
      refresh_note = simulate_refresh_interval ? " refresh_interval=#{simulate_refresh_interval}" : ""
      oracle_refresh_note = simulate_oracle_refresh_interval ? " oracle_refresh_interval=#{simulate_oracle_refresh_interval}" : ""
      puts "logit_drift_policy layers=#{simulate_logit_layers.join(',')} rank=#{rank} mean_cos=#{logit[:mean_cos].round(8)} min_cos=#{logit[:min_cos].round(8)} max_delta=#{logit[:max_delta].round(6)} top1_match=#{logit[:top1_match].round(2)}% top5_hit=#{logit[:top5_hit].round(2)}% mean_kl=#{logit[:mean_kl].round(8)} max_kl=#{logit[:max_kl].round(8)} min_margin=#{logit[:min_margin].round(6)} confident_mismatches=#{logit[:confident_mismatches]} approx_steps=#{logit[:approx_steps]} fallback_steps=#{logit[:fallback_steps]}#{fallback_note}#{refresh_note}#{oracle_refresh_note}#{output_note}"

      if simulate_generate_tokens > 0
        gen = simulate_greedy_policy(weights, token_ids, simulate_generate_tokens, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, simulate_oracle_refresh_interval, simulate_output_margin_threshold)
        gen_total_steps = gen[:approx_steps] + gen[:fallback_steps]
        gen_approx_rate = gen_total_steps > 0 ? (100.0 * gen[:approx_steps] / gen_total_steps) : 0.0
        gen_output_note = simulate_output_margin_threshold ? " output_margin_threshold=#{simulate_output_margin_threshold} output_fallbacks=#{gen[:output_fallbacks]}" : ""
        gen_refresh_note = simulate_refresh_interval ? " refresh_interval=#{simulate_refresh_interval}" : ""
        gen_oracle_refresh_note = simulate_oracle_refresh_interval ? " oracle_refresh_interval=#{simulate_oracle_refresh_interval}" : ""
        puts "greedy_drift_policy layers=#{simulate_logit_layers.join(',')} rank=#{rank} gen_tokens=#{simulate_generate_tokens} mean_cos=#{gen[:mean_cos].round(8)} min_cos=#{gen[:min_cos].round(8)} max_delta=#{gen[:max_delta].round(6)} top1_match=#{gen[:top1_match].round(2)}% top5_hit=#{gen[:top5_hit].round(2)}% mean_kl=#{gen[:mean_kl].round(8)} max_kl=#{gen[:max_kl].round(8)} min_margin=#{gen[:min_margin].round(6)} confident_mismatches=#{gen[:confident_mismatches]} approx_steps=#{gen[:approx_steps]} fallback_steps=#{gen[:fallback_steps]} approx_rate=#{gen_approx_rate.round(2)}%#{fallback_score_note}#{gen_refresh_note}#{gen_oracle_refresh_note}#{gen_output_note} exact_ids=#{gen[:exact_ids].join(',')} approx_ids=#{gen[:approx_ids].join(',')}"
      end

      if simulate_generate_tokens > 0 && !simulate_self_spec_gammas.empty?
        simulate_self_spec_gammas.each do |gamma|
          spec = simulate_self_spec_policy(weights, token_ids, simulate_generate_tokens, gamma, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, nil, nil, nil, simulate_self_spec_draft_margin, simulate_self_spec_draft_stop_margin, simulate_self_spec_topk_rescue)
          spec_total_steps = spec[:approx_steps] + spec[:fallback_steps]
          spec_approx_rate = spec_total_steps > 0 ? (100.0 * spec[:approx_steps] / spec_total_steps) : 0.0
          cost_note = ""
          if self_spec_cost_model
            estimated_cost = self_spec_estimated_cost(spec, self_spec_draft_cost, self_spec_verifier_cost,
              self_spec_chunk_overhead, self_spec_correction_cost, self_spec_overlap_cost, self_spec_overlap_efficiency)
            plain_cost = simulate_generate_tokens.to_f64
            estimated_speedup = estimated_cost > 0.0 ? plain_cost / estimated_cost : 0.0
            overlap_note = self_spec_overlap_cost ? ",overlap_eff:#{self_spec_overlap_efficiency.round(4)}" : ""
            cost_note = " cost_model=#{self_spec_overlap_cost ? "overlap" : "sum"}:draft:#{self_spec_draft_cost.round(4)},verifier:#{self_spec_verifier_cost.round(4)},chunk:#{self_spec_chunk_overhead.round(4)},correction:#{self_spec_correction_cost.round(4)}#{overlap_note} estimated_cost=#{estimated_cost.round(4)} estimated_speedup=#{estimated_speedup.round(4)}x"
          end
          rescue_note = simulate_self_spec_topk_rescue ? " topk_rescue=#{simulate_self_spec_topk_rescue} topk_rescues=#{spec[:topk_rescues]}" : ""
          puts "self_spec_policy layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{gamma} gen_tokens=#{simulate_generate_tokens} chunks=#{spec[:chunks]} full_accept_chunks=#{spec[:full_accept_chunks]} rejections=#{spec[:rejections]}#{rescue_note} accepted_draft_tokens=#{spec[:accepted_draft_tokens]} proposed_tokens=#{spec[:proposed_tokens]} accept_rate=#{spec[:accept_rate].round(2)}% avg_accept=#{spec[:avg_accept].round(3)} verifier_tokens=#{spec[:verifier_tokens]} correction_steps=#{spec[:correction_steps]} approx_steps=#{spec[:approx_steps]} fallback_steps=#{spec[:fallback_steps]} approx_rate=#{spec_approx_rate.round(2)}%#{fallback_score_note} draft_top2_hit=#{spec[:draft_top2_hit_rate].round(2)}% draft_top5_hit=#{spec[:draft_top5_hit_rate].round(2)}% reject_top2_hits=#{spec[:reject_top2_hits]} reject_top5_hits=#{spec[:reject_top5_hits]} break_even_draft_verify_per_proposed=#{spec[:break_even_draft_verify_per_proposed].round(4)} gamma_history=#{spec[:gamma_history].join(',')} draft_min_margin_history=#{spec[:draft_min_margin_history].map { |v| v.round(4) }.join(',')} draft_low_margin_history=#{spec[:draft_low_margin_history].join(',')}#{cost_note} exact_ids=#{spec[:exact_ids].join(',')} emitted_ids=#{spec[:emitted_ids].join(',')}"
        end
      end
      if simulate_generate_tokens > 0 && simulate_self_spec_adaptive
        spec = simulate_self_spec_policy(weights, token_ids, simulate_generate_tokens, simulate_self_spec_adaptive_start, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, simulate_self_spec_adaptive_min, simulate_self_spec_adaptive_max, simulate_self_spec_adaptive_grow_margin, simulate_self_spec_draft_margin, simulate_self_spec_draft_stop_margin, simulate_self_spec_topk_rescue)
        spec_total_steps = spec[:approx_steps] + spec[:fallback_steps]
        spec_approx_rate = spec_total_steps > 0 ? (100.0 * spec[:approx_steps] / spec_total_steps) : 0.0
        cost_note = ""
        if self_spec_cost_model
          estimated_cost = self_spec_estimated_cost(spec, self_spec_draft_cost, self_spec_verifier_cost,
            self_spec_chunk_overhead, self_spec_correction_cost, self_spec_overlap_cost, self_spec_overlap_efficiency)
          plain_cost = simulate_generate_tokens.to_f64
          estimated_speedup = estimated_cost > 0.0 ? plain_cost / estimated_cost : 0.0
          overlap_note = self_spec_overlap_cost ? ",overlap_eff:#{self_spec_overlap_efficiency.round(4)}" : ""
          cost_note = " cost_model=#{self_spec_overlap_cost ? "overlap" : "sum"}:draft:#{self_spec_draft_cost.round(4)},verifier:#{self_spec_verifier_cost.round(4)},chunk:#{self_spec_chunk_overhead.round(4)},correction:#{self_spec_correction_cost.round(4)}#{overlap_note} estimated_cost=#{estimated_cost.round(4)} estimated_speedup=#{estimated_speedup.round(4)}x"
        end
        grow_margin_note = simulate_self_spec_adaptive_grow_margin ? " grow_margin=#{simulate_self_spec_adaptive_grow_margin}" : ""
        rescue_note = simulate_self_spec_topk_rescue ? " topk_rescue=#{simulate_self_spec_topk_rescue} topk_rescues=#{spec[:topk_rescues]}" : ""
        puts "self_spec_adaptive layers=#{simulate_logit_layers.join(',')} rank=#{rank} min_gamma=#{simulate_self_spec_adaptive_min} start_gamma=#{simulate_self_spec_adaptive_start} max_gamma=#{simulate_self_spec_adaptive_max}#{grow_margin_note} gen_tokens=#{simulate_generate_tokens} chunks=#{spec[:chunks]} full_accept_chunks=#{spec[:full_accept_chunks]} rejections=#{spec[:rejections]}#{rescue_note} accepted_draft_tokens=#{spec[:accepted_draft_tokens]} proposed_tokens=#{spec[:proposed_tokens]} accept_rate=#{spec[:accept_rate].round(2)}% avg_accept=#{spec[:avg_accept].round(3)} verifier_tokens=#{spec[:verifier_tokens]} correction_steps=#{spec[:correction_steps]} approx_steps=#{spec[:approx_steps]} fallback_steps=#{spec[:fallback_steps]} approx_rate=#{spec_approx_rate.round(2)}%#{fallback_score_note} draft_top2_hit=#{spec[:draft_top2_hit_rate].round(2)}% draft_top5_hit=#{spec[:draft_top5_hit_rate].round(2)}% reject_top2_hits=#{spec[:reject_top2_hits]} reject_top5_hits=#{spec[:reject_top5_hits]} break_even_draft_verify_per_proposed=#{spec[:break_even_draft_verify_per_proposed].round(4)} gamma_history=#{spec[:gamma_history].join(',')} draft_min_margin_history=#{spec[:draft_min_margin_history].map { |v| v.round(4) }.join(',')} draft_low_margin_history=#{spec[:draft_low_margin_history].join(',')}#{cost_note} exact_ids=#{spec[:exact_ids].join(',')} emitted_ids=#{spec[:emitted_ids].join(',')}"
      end
      if simulate_generate_tokens > 0 && !simulate_self_spec_progressive.empty?
        spec = simulate_self_spec_policy(weights, token_ids, simulate_generate_tokens, simulate_self_spec_progressive[0], layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, nil, nil, nil, simulate_self_spec_draft_margin, simulate_self_spec_draft_stop_margin, simulate_self_spec_topk_rescue, simulate_self_spec_progressive)
        spec_total_steps = spec[:approx_steps] + spec[:fallback_steps]
        spec_approx_rate = spec_total_steps > 0 ? (100.0 * spec[:approx_steps] / spec_total_steps) : 0.0
        cost_note = ""
        if self_spec_cost_model
          estimated_cost = self_spec_estimated_cost(spec, self_spec_draft_cost, self_spec_verifier_cost,
            self_spec_chunk_overhead, self_spec_correction_cost, self_spec_overlap_cost, self_spec_overlap_efficiency)
          plain_cost = simulate_generate_tokens.to_f64
          estimated_speedup = estimated_cost > 0.0 ? plain_cost / estimated_cost : 0.0
          overlap_note = self_spec_overlap_cost ? ",overlap_eff:#{self_spec_overlap_efficiency.round(4)}" : ""
          cost_note = " cost_model=#{self_spec_overlap_cost ? "overlap" : "sum"}:draft:#{self_spec_draft_cost.round(4)},verifier:#{self_spec_verifier_cost.round(4)},chunk:#{self_spec_chunk_overhead.round(4)},correction:#{self_spec_correction_cost.round(4)}#{overlap_note} estimated_cost=#{estimated_cost.round(4)} estimated_speedup=#{estimated_speedup.round(4)}x"
        end
        rescue_note = simulate_self_spec_topk_rescue ? " topk_rescue=#{simulate_self_spec_topk_rescue} topk_rescues=#{spec[:topk_rescues]}" : ""
        puts "self_spec_progressive layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{simulate_self_spec_progressive.join(',')} gen_tokens=#{simulate_generate_tokens} chunks=#{spec[:chunks]} full_accept_chunks=#{spec[:full_accept_chunks]} rejections=#{spec[:rejections]}#{rescue_note} accepted_draft_tokens=#{spec[:accepted_draft_tokens]} proposed_tokens=#{spec[:proposed_tokens]} accept_rate=#{spec[:accept_rate].round(2)}% avg_accept=#{spec[:avg_accept].round(3)} verifier_tokens=#{spec[:verifier_tokens]} correction_steps=#{spec[:correction_steps]} approx_steps=#{spec[:approx_steps]} fallback_steps=#{spec[:fallback_steps]} approx_rate=#{spec_approx_rate.round(2)}%#{fallback_score_note} draft_top2_hit=#{spec[:draft_top2_hit_rate].round(2)}% draft_top5_hit=#{spec[:draft_top5_hit_rate].round(2)}% reject_top2_hits=#{spec[:reject_top2_hits]} reject_top5_hits=#{spec[:reject_top5_hits]} break_even_draft_verify_per_proposed=#{spec[:break_even_draft_verify_per_proposed].round(4)} gamma_history=#{spec[:gamma_history].join(',')} draft_min_margin_history=#{spec[:draft_min_margin_history].map { |v| v.round(4) }.join(',')} draft_low_margin_history=#{spec[:draft_low_margin_history].join(',')}#{cost_note} exact_ids=#{spec[:exact_ids].join(',')} emitted_ids=#{spec[:emitted_ids].join(',')}"
      end
      if simulate_generate_tokens > 0 && (tree_k = simulate_self_spec_tree_k)
        tree_schedule = simulate_self_spec_progressive.empty? ? [2, 2, 4] : simulate_self_spec_progressive
        tree = simulate_self_spec_tree_oracle(weights, token_ids, simulate_generate_tokens, tree_k, tree_schedule, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval)
        tree_total_steps = tree[:approx_steps] + tree[:fallback_steps]
        tree_approx_rate = tree_total_steps > 0 ? (100.0 * tree[:approx_steps] / tree_total_steps) : 0.0
        parity = tree[:exact_ids] == tree[:emitted_ids]
        tree_cost_note = ""
        if self_spec_cost_model
          rank_cost = self_spec_tree_estimated_cost(tree, self_spec_draft_cost, self_spec_verifier_cost, self_spec_chunk_overhead, self_spec_correction_cost, tree[:branch_tokens_rank])
          full_cost = self_spec_tree_estimated_cost(tree, self_spec_draft_cost, self_spec_verifier_cost, self_spec_chunk_overhead, self_spec_correction_cost, tree[:branch_tokens_full])
          tree_cost_note = " cost_model=tree:draft:#{self_spec_draft_cost.round(4)},verifier:#{self_spec_verifier_cost.round(4)},chunk:#{self_spec_chunk_overhead.round(4)},correction:#{self_spec_correction_cost.round(4)} rank_cost=#{rank_cost.round(4)} rank_speedup=#{(simulate_generate_tokens / rank_cost).round(4)}x full_cost=#{full_cost.round(4)} full_speedup=#{(simulate_generate_tokens / full_cost).round(4)}x"
        end
        puts "self_spec_tree_oracle layers=#{simulate_logit_layers.join(',')} rank=#{rank} top_k=#{tree_k} schedule=#{tree_schedule.join(',')} gen_tokens=#{simulate_generate_tokens} chunks=#{tree[:chunks]} full_rescue_chunks=#{tree[:full_rescue_chunks]} misses=#{tree[:misses]} parity=#{parity} draft_steps=#{tree[:draft_steps]} top1_hits=#{tree[:top1_hits]} topk_hits=#{tree[:topk_hits]} top1_rate=#{tree[:top1_rate].round(2)}% topk_rate=#{tree[:topk_rate].round(2)}% branch_tokens_rank=#{tree[:branch_tokens_rank]} branch_tokens_full=#{tree[:branch_tokens_full]} avg_rank_branch_tokens=#{tree[:avg_rank_branch_tokens].round(3)} avg_full_branch_tokens=#{tree[:avg_full_branch_tokens].round(3)} correction_steps=#{tree[:correction_steps]} approx_steps=#{tree[:approx_steps]} fallback_steps=#{tree[:fallback_steps]} approx_rate=#{tree_approx_rate.round(2)}%#{fallback_score_note} schedule_history=#{tree[:schedule_history].join(',')}#{tree_cost_note} exact_ids=#{tree[:exact_ids].join(',')} emitted_ids=#{tree[:emitted_ids].join(',')}"
      end
      if simulate_generate_tokens > 0 && (oracle_k = simulate_topk_oracle_k)
        oracle = simulate_topk_oracle_calibration(weights, token_ids, simulate_generate_tokens, oracle_k, simulate_topk_oracle_train_tokens, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval)
        delta_branch = oracle[:baseline_avg_branch_tokens] - oracle[:calibrated_avg_branch_tokens]
        puts "topk_oracle_calibration layers=#{simulate_logit_layers.join(',')} rank=#{rank} top_k=#{oracle_k} gen_tokens=#{simulate_generate_tokens} samples=#{oracle[:samples]} train=#{oracle[:train_samples]} test=#{oracle[:test_samples]} best_token_scale=#{oracle[:best_token_scale]} best_rank_scale=#{oracle[:best_rank_scale]} best_margin_threshold=#{oracle[:best_margin_threshold].round(4)} train_top1=#{oracle[:train_top1_rate].round(2)}% train_topk=#{oracle[:train_topk_rate].round(2)}% train_avg_branch=#{oracle[:train_avg_branch_tokens].round(3)} baseline_top1=#{oracle[:baseline_top1_rate].round(2)}% baseline_topk=#{oracle[:baseline_topk_rate].round(2)}% baseline_avg_branch=#{oracle[:baseline_avg_branch_tokens].round(3)} baseline_misses=#{oracle[:baseline_misses]} calibrated_top1=#{oracle[:calibrated_top1_rate].round(2)}% calibrated_topk=#{oracle[:calibrated_topk_rate].round(2)}% calibrated_avg_branch=#{oracle[:calibrated_avg_branch_tokens].round(3)} calibrated_misses=#{oracle[:calibrated_misses]} delta_avg_branch=#{delta_branch.round(3)} margin_gate_rate=#{oracle[:margin_gate_rate].round(2)}% margin_gate_topk=#{oracle[:margin_gate_topk_rate].round(2)}% margin_gate_avg_branch=#{oracle[:margin_gate_avg_branch_tokens].round(3)} margin_gate_misses=#{oracle[:margin_gate_misses]} margin_gate_cost=#{oracle[:margin_gate_cost].round(3)} exact_ids=#{oracle[:exact_ids].join(',')}"
      end
      if simulate_generate_tokens > 0 && !simulate_self_spec_wall_progressive.empty?
        wall = simulate_self_spec_wall_policy(weights, token_ids, simulate_generate_tokens, simulate_self_spec_wall_progressive, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, simulate_self_spec_wall_metal_lowrank, simulate_self_spec_wall_metal_project, simulate_self_spec_wall_metal_layer_updown)
        metal_note = simulate_self_spec_wall_metal_layer_updown ? " metal_project=1 metal_layer_updown=1" : (simulate_self_spec_wall_metal_project ? " metal_project=1" : (simulate_self_spec_wall_metal_lowrank ? " metal_lowrank=1" : ""))
        puts "self_spec_wall_progressive layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{simulate_self_spec_wall_progressive.join(',')}#{metal_note} gen_tokens=#{simulate_generate_tokens} chunks=#{wall[:chunks]} rejections=#{wall[:rejections]} accepted_draft_tokens=#{wall[:accepted_draft_tokens]} proposed_tokens=#{wall[:proposed_tokens]} accept_rate=#{wall[:accept_rate].round(2)}% verifier_tokens=#{wall[:verifier_tokens]} correction_steps=#{wall[:correction_steps]} draft_ms=#{wall[:draft_ms].round(3)} verifier_ms=#{wall[:verifier_ms].round(3)} replay_ms=#{wall[:replay_ms].round(3)} serial_ms=#{wall[:serial_ms].round(3)} overlap_est_ms=#{wall[:overlap_est_ms].round(3)} speedup_est=#{wall[:speedup_est].round(4)}x exact_ids=#{wall[:exact_ids].join(',')} emitted_ids=#{wall[:emitted_ids].join(',')}"
      end
      if simulate_generate_tokens > 0 && !simulate_self_spec_wall_progressive.empty? && !simulate_cheap_self_draft_variants.empty?
        simulate_cheap_self_draft_variants.each do |variant|
          unless cheap_draft_variant_valid?(variant)
            raise "unknown cheap self-draft variant #{variant.inspect}"
          end
          wall = simulate_self_spec_wall_policy(weights, token_ids, simulate_generate_tokens, simulate_self_spec_wall_progressive, layer_bases, rank, calib_count, fallback_threshold, simulate_refresh_interval, simulate_self_spec_wall_metal_lowrank, simulate_self_spec_wall_metal_project, simulate_self_spec_wall_metal_layer_updown, variant, ffn_activation_bases, ffn_down_adapters, ffn_updown_adapters, ffn_block_selectors)
          metal_note = simulate_self_spec_wall_metal_layer_updown ? " metal_project=1 metal_layer_updown=1" : (simulate_self_spec_wall_metal_project ? " metal_project=1" : (simulate_self_spec_wall_metal_lowrank ? " metal_lowrank=1" : ""))
          parity = wall[:exact_ids] == wall[:emitted_ids]
          puts "cheap_self_draft_variant=#{variant} layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{simulate_self_spec_wall_progressive.join(',')}#{metal_note} gen_tokens=#{simulate_generate_tokens} chunks=#{wall[:chunks]} rejections=#{wall[:rejections]} accepted_draft_tokens=#{wall[:accepted_draft_tokens]} proposed_tokens=#{wall[:proposed_tokens]} accept_rate=#{wall[:accept_rate].round(2)}% parity=#{parity} verifier_tokens=#{wall[:verifier_tokens]} correction_steps=#{wall[:correction_steps]} draft_ms=#{wall[:draft_ms].round(3)} verifier_ms=#{wall[:verifier_ms].round(3)} replay_ms=#{wall[:replay_ms].round(3)} serial_ms=#{wall[:serial_ms].round(3)} overlap_est_ms=#{wall[:overlap_est_ms].round(3)} speedup_est=#{wall[:speedup_est].round(4)}x exact_ids=#{wall[:exact_ids].join(',')} emitted_ids=#{wall[:emitted_ids].join(',')}"
        end
      end
    end
    if simulate_lowrank_eval_suite
      suite_token_sets = [{name: "main", token_ids: token_ids}] of PromptTokenSet
      simulate_lowrank_eval_suite_prompts.each do |suite_prompt|
        suite_token_sets << {
          name:      suite_prompt[:name],
          token_ids: token_ids_for_prompt(tok, suite_prompt[:text], tokens_limit),
        }
      end
      suite_thresholds = if simulate_fallback_thresholds.empty?
                           [simulate_fallback_threshold]
                         else
                           simulate_fallback_thresholds.map { |v| v.as(Float64?) }
                         end
      run_lowrank_eval_suite(weights, suite_token_sets, sorted_simulate_logit_layers, rank, max_rank,
        basis_mode, pca_iters, calib_tokens, simulate_generate_tokens, suite_thresholds,
        simulate_refresh_interval, simulate_oracle_refresh_interval, simulate_output_margin_threshold,
        simulate_self_spec_gammas, simulate_self_spec_draft_margin, simulate_self_spec_draft_stop_margin,
        simulate_self_spec_topk_rescue)
    end
    if simulate_self_draft_metal_baseline > 0
      sd = simulate_self_draft_metal_baseline_run(weights, token_ids, calib_count, simulate_self_draft_metal_baseline, layer_bases, rank)
      puts "self_draft_metal_baseline layers=#{simulate_logit_layers.join(',')} rank=#{rank} steps=#{sd[:steps]} self_draft_ms=#{sd[:self_draft_ms].round(3)} exact_ms=#{sd[:exact_ms].round(3)} verifier_ms=#{sd[:verifier_ms].round(3)} self_draft_per_tok_ms=#{sd[:self_draft_per_token_ms].round(3)} exact_per_tok_ms=#{sd[:exact_per_token_ms].round(3)} verifier_per_tok_ms=#{sd[:verifier_per_token_ms].round(3)} self_spec_wall_ratio=#{sd[:self_spec_wall_ratio].round(4)} agreement=#{sd[:agreement]}/#{sd[:steps]} self_draft_ids=#{sd[:self_draft_ids].join(',')} exact_ids=#{sd[:exact_ids].join(',')} verifier_ids=#{sd[:verifier_ids].join(',')}"
    end
    if simulate_self_draft_gpu_chain > 0
      chain = simulate_self_draft_gpu_chain_run(weights, token_ids, calib_count, simulate_self_draft_gpu_chain, layer_bases, rank)
      puts "self_draft_gpu_chain layers=#{simulate_logit_layers.join(',')} rank=#{rank} steps=#{chain[:steps]} submit_ms=#{chain[:submit_ms].round(3)} wait_ms=#{chain[:wait_ms].round(3)} chain_ms=#{chain[:chain_ms].round(3)} exact_ms=#{chain[:exact_ms].round(3)} agreement=#{chain[:agreement]}/#{chain[:steps]} chain_ids=#{chain[:chain_ids].join(',')} exact_ids=#{chain[:exact_ids].join(',')}"
      if simulate_self_draft_gpu_chain_text
        puts "self_draft_gpu_chain_text layers=#{simulate_logit_layers.join(',')} rank=#{rank} steps=#{chain[:steps]} agreement=#{chain[:agreement]}/#{chain[:steps]} draft_text=#{tok.decode(chain[:chain_ids]).inspect} exact_text=#{tok.decode(chain[:exact_ids]).inspect}"
      end
      if chain_updown_rank = simulate_self_draft_gpu_chain_updown_rank
        chain_updown_layer_set = simulate_self_draft_gpu_chain_updown_layers.empty? ? nil : Set(Int32).new(simulate_self_draft_gpu_chain_updown_layers)
        updown_chain = simulate_self_draft_gpu_chain_run(weights, token_ids, calib_count, simulate_self_draft_gpu_chain, layer_bases, rank,
          chain_updown_rank, ffn_updown_adapters, chain_updown_layer_set)
        updown_layer_note = chain_updown_layer_set ? " updown_layers=#{chain_updown_layer_set.not_nil!.to_a.sort.join(',')}" : ""
        puts "self_draft_gpu_chain_updown layers=#{simulate_logit_layers.join(',')} rank=#{rank} updown_rank=#{updown_chain[:updown_rank]}#{updown_layer_note} steps=#{updown_chain[:steps]} submit_ms=#{updown_chain[:submit_ms].round(3)} wait_ms=#{updown_chain[:wait_ms].round(3)} chain_ms=#{updown_chain[:chain_ms].round(3)} exact_ms=#{updown_chain[:exact_ms].round(3)} agreement=#{updown_chain[:agreement]}/#{updown_chain[:steps]} chain_ids=#{updown_chain[:chain_ids].join(',')} exact_ids=#{updown_chain[:exact_ids].join(',')}"
        if simulate_self_draft_gpu_chain_text
          puts "self_draft_gpu_chain_updown_text layers=#{simulate_logit_layers.join(',')} rank=#{rank} updown_rank=#{updown_chain[:updown_rank]}#{updown_layer_note} steps=#{updown_chain[:steps]} agreement=#{updown_chain[:agreement]}/#{updown_chain[:steps]} draft_text=#{tok.decode(updown_chain[:chain_ids]).inspect} exact_text=#{tok.decode(updown_chain[:exact_ids]).inspect}"
        end
      end
    end
    if simulate_self_draft_gpu_state_only > 0
      state_only = simulate_self_draft_gpu_state_only_run(weights, token_ids, calib_count, simulate_self_draft_gpu_state_only, layer_bases, rank)
      puts "self_draft_gpu_state_only layers=#{simulate_logit_layers.join(',')} rank=#{rank} steps=#{state_only[:steps]} project_ms=#{state_only[:project_ms].round(3)} submit_ms=#{state_only[:submit_ms].round(3)} wait_ms=#{state_only[:wait_ms].round(3)} chain_ms=#{state_only[:chain_ms].round(3)} per_token_ms=#{state_only[:per_token_ms].round(3)}"
    end
    if simulate_self_draft_gpu_chain_overlap > 0
      ov = simulate_self_draft_gpu_chain_overlap_run(weights, token_ids, calib_count, simulate_self_draft_gpu_chain_overlap, layer_bases, rank)
      puts "self_draft_gpu_chain_overlap layers=#{simulate_logit_layers.join(',')} rank=#{rank} steps=#{ov[:steps]} draft_alone_ms=#{ov[:draft_alone_ms].round(3)} verifier_ms=#{ov[:verifier_ms].round(3)} overlap_ms=#{ov[:overlap_ms].round(3)} draft_submit_ms=#{ov[:draft_submit_ms].round(3)} draft_wait_ms=#{ov[:draft_wait_ms].round(3)} hidden_ms=#{ov[:hidden_ms].round(3)} speedup=#{ov[:speedup].round(4)} agreement=#{ov[:agreement]}/#{ov[:steps]} draft_ids=#{ov[:draft_ids].join(',')} exact_ids=#{ov[:exact_ids].join(',')} verifier_ids=#{ov[:verifier_ids].join(',')}"
    end
    if simulate_mtp_self_draft_fusion > 0
      mtp = ML::GGUF::Qwen35MTPWeights.from_safetensors(mtp_path)
      mtp.validate_for_qwen35!(weights.hparams)
      fusion_updown_layer_set = simulate_mtp_self_draft_fusion_updown_layers.empty? ? nil : Set(Int32).new(simulate_mtp_self_draft_fusion_updown_layers)
      fusion = simulate_mtp_self_draft_fusion_run(weights, mtp, token_ids, calib_count, simulate_mtp_self_draft_fusion,
        layer_bases, rank, simulate_mtp_self_draft_fusion_topk, simulate_mtp_self_draft_fusion_updown_rank,
        ffn_updown_adapters, fusion_updown_layer_set)
      steps = fusion[:steps]
      self_first_avg = fusion[:self_first_attempts_total].to_f64 / steps
      mtp_first_avg = fusion[:mtp_first_attempts_total].to_f64 / steps
      mtp_extra_hits = fusion[:union_hits] - fusion[:self_hits]
      updown_note = fusion[:draft_updown_rank] > 0 ? " draft_pca_updown_rank=#{fusion[:draft_updown_rank]} draft_pca_updown_layers=#{simulate_mtp_self_draft_fusion_updown_layers.empty? ? "all" : simulate_mtp_self_draft_fusion_updown_layers.join(',')}" : ""
      puts "mtp_self_draft_fusion layers=#{simulate_logit_layers.join(',')} rank=#{rank}#{updown_note} steps=#{steps} mtp_topk=#{fusion[:mtp_topk]} self_hits=#{fusion[:self_hits]} self_rate=#{(100.0 * fusion[:self_hits] / steps).round(2)} mtp_hits=#{fusion[:mtp_hits]} mtp_rate=#{(100.0 * fusion[:mtp_hits] / steps).round(2)} union_hits=#{fusion[:union_hits]} union_rate=#{(100.0 * fusion[:union_hits] / steps).round(2)} mtp_extra_hits=#{mtp_extra_hits} self_only_hits=#{fusion[:additive_hits]} agreement=#{fusion[:agreement]} agreement_hits=#{fusion[:agreement_hits]} agreement_false=#{fusion[:agreement_false]} self_first_attempts=#{fusion[:self_first_attempts_total]} self_first_avg_attempts=#{self_first_avg.round(3)} mtp_first_attempts=#{fusion[:mtp_first_attempts_total]} mtp_first_avg_attempts=#{mtp_first_avg.round(3)} self_chain_ms=#{fusion[:self_chain_ms].round(3)} mtp_ms=#{fusion[:mtp_ms].round(3)} exact_ms=#{fusion[:self_exact_ms].round(3)} self_ids=#{fusion[:self_ids].join(',')} exact_ids=#{fusion[:exact_ids].join(',')}"
      fusion_rows = fusion[:rows]
      self_misses = steps - fusion[:self_hits]
      mtp_rescue_hits = fusion_rows.count { |row| !row[:self_hit] && row[:mtp_hit] }
      mtp_unresolved_misses = fusion_rows.count { |row| !row[:self_hit] && !row[:mtp_hit] }
      avg_union_k2_width = fusion[:union_k2_size_total].to_f64 / steps
      mtp_k2_width = Math.min(2, fusion[:mtp_topk])
      self_top2_attempts_total = fusion_rows.sum { |row| row[:self_hit] ? 1 : 2 }
      self_top2_unresolved = steps - fusion[:self_top2_hits]
      self_top2_mtp_k2_attempts_total = fusion_rows.sum do |row|
        if row[:self_hit]
          1
        elsif row[:self_top2_hit]
          2
        else
          2 + (row[:mtp_k2_hit] ? row[:mtp_rank] : mtp_k2_width)
        end
      end
      self_top2_mtp_topk_attempts_total = fusion_rows.sum do |row|
        if row[:self_hit]
          1
        elsif row[:self_top2_hit]
          2
        else
          2 + (row[:mtp_hit] ? row[:mtp_rank] : fusion[:mtp_topk])
        end
      end
      self_top2_mtp_k2_rescues = fusion_rows.count { |row| !row[:self_top2_hit] && row[:mtp_k2_hit] }
      self_top2_mtp_topk_rescues = fusion_rows.count { |row| !row[:self_top2_hit] && row[:mtp_hit] }
      mismatch_mtp_ms = fusion[:mtp_ms] * self_misses.to_f64 / steps
      self_top2_mismatch_mtp_ms = fusion[:mtp_ms] * self_top2_unresolved.to_f64 / steps
      mismatch_saved_ms = fusion[:mtp_ms] - mismatch_mtp_ms
      self_top2_mismatch_saved_ms = fusion[:mtp_ms] - self_top2_mismatch_mtp_ms
      agreement_selected = fusion[:agreement]
      agreement_fallback = steps - agreement_selected
      puts "mtp_self_draft_hybrid_policy policy=self_first_mtp_on_miss layers=#{simulate_logit_layers.join(',')} rank=#{rank}#{updown_note} steps=#{steps} mtp_topk=#{fusion[:mtp_topk]} self_misses=#{self_misses} mtp_calls=#{self_misses} mtp_call_rate=#{pct_count(self_misses, steps).round(2)} mtp_rescue_hits=#{mtp_rescue_hits} mtp_rescue_rate=#{pct_count(mtp_rescue_hits, self_misses).round(2)} unresolved_misses=#{mtp_unresolved_misses} self_first_attempts=#{fusion[:self_first_attempts_total]} self_first_avg_attempts=#{self_first_avg.round(3)} modeled_mtp_ms=#{mismatch_mtp_ms.round(3)} saved_mtp_ms=#{mismatch_saved_ms.round(3)} note=oracle_accounting_exact_positions_not_resyncing_wall_runtime"
      puts "mtp_self_draft_hybrid_policy policy=agreement_guard layers=#{simulate_logit_layers.join(',')} rank=#{rank}#{updown_note} steps=#{steps} mtp_topk=#{fusion[:mtp_topk]} mtp_calls=#{steps} selected=#{agreement_selected} selected_rate=#{pct_count(agreement_selected, steps).round(2)} selected_hits=#{fusion[:agreement_hits]} selected_false=#{fusion[:agreement_false]} precision=#{pct_count(fusion[:agreement_hits], agreement_selected).round(2)} fallback=#{agreement_fallback} fallback_rate=#{pct_count(agreement_fallback, steps).round(2)} note=agreement_requires_always_on_mtp_unless_used_as_calibration_feature"
      puts "mtp_self_draft_candidate_union layers=#{simulate_logit_layers.join(',')} rank=#{rank}#{updown_note} steps=#{steps} mtp_topk=#{fusion[:mtp_topk]} self_top2_hits=#{fusion[:self_top2_hits]} self_top2_rate=#{pct_count(fusion[:self_top2_hits], steps).round(2)} mtp_k2_hits=#{fusion[:mtp_k2_hits]} mtp_k2_rate=#{pct_count(fusion[:mtp_k2_hits], steps).round(2)} union_k2_hits=#{fusion[:union_k2_hits]} union_k2_rate=#{pct_count(fusion[:union_k2_hits], steps).round(2)} union_topk_hits=#{fusion[:union_topk_with_self_top2_hits]} union_topk_rate=#{pct_count(fusion[:union_topk_with_self_top2_hits], steps).round(2)} mtp_extra_over_self_top2=#{fusion[:mtp_extra_over_self_top2]} mtp_extra_over_self_top2_topk=#{fusion[:mtp_extra_over_self_top2_topk]} self_top2_extra_over_mtp_k2=#{fusion[:self_top2_extra_over_mtp_k2]} self_top2_extra_over_mtp_topk=#{fusion[:self_top2_extra_over_mtp_topk]} both_top2_hits=#{fusion[:both_top2_hits]} avg_union_k2_width=#{avg_union_k2_width.round(3)} note=oracle_accounting_compares_existing_self_top2_against_mtp_top2_and_topk"
      puts "mtp_self_draft_route_policy policy=self_top2_only layers=#{simulate_logit_layers.join(',')} rank=#{rank}#{updown_note} steps=#{steps} resolved_hits=#{fusion[:self_top2_hits]} resolved_rate=#{pct_count(fusion[:self_top2_hits], steps).round(2)} unresolved=#{self_top2_unresolved} attempts=#{self_top2_attempts_total} avg_attempts=#{(self_top2_attempts_total.to_f64 / steps).round(3)} mtp_calls=0 modeled_mtp_ms=0.0 note=baseline_existing_tree2_candidate_pressure"
      puts "mtp_self_draft_route_policy policy=self_top2_first_mtp_k2_on_miss layers=#{simulate_logit_layers.join(',')} rank=#{rank}#{updown_note} steps=#{steps} mtp_width=#{mtp_k2_width} resolved_hits=#{fusion[:union_k2_hits]} resolved_rate=#{pct_count(fusion[:union_k2_hits], steps).round(2)} unresolved=#{steps - fusion[:union_k2_hits]} mtp_calls=#{self_top2_unresolved} mtp_call_rate=#{pct_count(self_top2_unresolved, steps).round(2)} mtp_rescue_hits=#{self_top2_mtp_k2_rescues} mtp_rescue_rate=#{pct_count(self_top2_mtp_k2_rescues, self_top2_unresolved).round(2)} attempts=#{self_top2_mtp_k2_attempts_total} avg_attempts=#{(self_top2_mtp_k2_attempts_total.to_f64 / steps).round(3)} modeled_mtp_ms=#{self_top2_mismatch_mtp_ms.round(3)} saved_mtp_ms=#{self_top2_mismatch_saved_ms.round(3)} note=oracle_policy_pressure_exact_positions_not_resyncing_wall_runtime"
      puts "mtp_self_draft_route_policy policy=self_top2_first_mtp_topk_on_miss layers=#{simulate_logit_layers.join(',')} rank=#{rank}#{updown_note} steps=#{steps} mtp_width=#{fusion[:mtp_topk]} resolved_hits=#{fusion[:union_topk_with_self_top2_hits]} resolved_rate=#{pct_count(fusion[:union_topk_with_self_top2_hits], steps).round(2)} unresolved=#{steps - fusion[:union_topk_with_self_top2_hits]} mtp_calls=#{self_top2_unresolved} mtp_call_rate=#{pct_count(self_top2_unresolved, steps).round(2)} mtp_rescue_hits=#{self_top2_mtp_topk_rescues} mtp_rescue_rate=#{pct_count(self_top2_mtp_topk_rescues, self_top2_unresolved).round(2)} attempts=#{self_top2_mtp_topk_attempts_total} avg_attempts=#{(self_top2_mtp_topk_attempts_total.to_f64 / steps).round(3)} modeled_mtp_ms=#{self_top2_mismatch_mtp_ms.round(3)} saved_mtp_ms=#{self_top2_mismatch_saved_ms.round(3)} note=oracle_policy_pressure_exact_positions_not_resyncing_wall_runtime"
      fusion[:rows].each do |row|
        puts "mtp_self_draft_fusion_step i=#{row[:index]} exact=#{row[:exact]} self=#{row[:self_id]} self_second=#{row[:self_second_id]} mtp_rank=#{row[:mtp_rank]} self_hit=#{row[:self_hit]} self_top2_hit=#{row[:self_top2_hit]} mtp_hit=#{row[:mtp_hit]} mtp_k2_hit=#{row[:mtp_k2_hit]} union_hit=#{row[:union_hit]} union_k2_hit=#{row[:union_k2_hit]} agreement=#{row[:agreement]} union_size=#{row[:union_size]} union_k2_size=#{row[:union_k2_size]} self_first_attempts=#{row[:self_first_attempts]} mtp_first_attempts=#{row[:mtp_first_attempts]}"
      end
      simulate_mtp_self_draft_fusion_suite_prompts.each do |suite_prompt|
        suite_token_ids = token_ids_for_prompt(tok, suite_prompt[:text], tokens_limit)
        suite_calib_count = Math.min(calib_tokens, suite_token_ids.size - 1)
        raise "MTP/self-draft fusion suite prompt #{suite_prompt[:name]} needs at least one held-out token" unless suite_calib_count > 0 && suite_calib_count < suite_token_ids.size
        suite_layer_bases = {} of Int32 => BasisSet
        sorted_simulate_logit_layers.each do |il|
          vectors = recurrent_k_vectors_for_prompt(weights, suite_token_ids, il)
          suite_layer_bases[il] = vectors.map do |head_vectors|
            build_basis(head_vectors[0, suite_calib_count], max_rank, basis_mode, pca_iters)
          end
        end
        suite_fusion = simulate_mtp_self_draft_fusion_run(weights, mtp, suite_token_ids, suite_calib_count, simulate_mtp_self_draft_fusion,
          suite_layer_bases, rank, simulate_mtp_self_draft_fusion_topk, simulate_mtp_self_draft_fusion_updown_rank,
          ffn_updown_adapters, fusion_updown_layer_set)
        suite_steps = suite_fusion[:steps]
        suite_rows = suite_fusion[:rows]
        suite_mtp_extra_hits = suite_fusion[:union_hits] - suite_fusion[:self_hits]
        suite_self_top2_unresolved = suite_steps - suite_fusion[:self_top2_hits]
        suite_mtp_k2_width = Math.min(2, suite_fusion[:mtp_topk])
        suite_self_top2_attempts_total = suite_rows.sum { |row| row[:self_hit] ? 1 : 2 }
        suite_self_top2_mtp_k2_attempts_total = suite_rows.sum do |row|
          if row[:self_hit]
            1
          elsif row[:self_top2_hit]
            2
          else
            2 + (row[:mtp_k2_hit] ? row[:mtp_rank] : suite_mtp_k2_width)
          end
        end
        suite_self_top2_mtp_k2_rescues = suite_rows.count { |row| !row[:self_top2_hit] && row[:mtp_k2_hit] }
        suite_self_top2_mismatch_mtp_ms = suite_fusion[:mtp_ms] * suite_self_top2_unresolved.to_f64 / suite_steps
        suite_self_top2_mismatch_saved_ms = suite_fusion[:mtp_ms] - suite_self_top2_mismatch_mtp_ms
        suite_updown_note = suite_fusion[:draft_updown_rank] > 0 ? " draft_pca_updown_rank=#{suite_fusion[:draft_updown_rank]} draft_pca_updown_layers=#{simulate_mtp_self_draft_fusion_updown_layers.empty? ? "all" : simulate_mtp_self_draft_fusion_updown_layers.join(',')}" : ""
        puts "mtp_self_draft_fusion_suite name=#{suite_prompt[:name]} layers=#{simulate_logit_layers.join(',')} rank=#{rank}#{suite_updown_note} steps=#{suite_steps} mtp_topk=#{suite_fusion[:mtp_topk]} self_hits=#{suite_fusion[:self_hits]} self_rate=#{pct_count(suite_fusion[:self_hits], suite_steps).round(2)} mtp_hits=#{suite_fusion[:mtp_hits]} mtp_rate=#{pct_count(suite_fusion[:mtp_hits], suite_steps).round(2)} union_hits=#{suite_fusion[:union_hits]} union_rate=#{pct_count(suite_fusion[:union_hits], suite_steps).round(2)} mtp_extra_hits=#{suite_mtp_extra_hits} self_top2_hits=#{suite_fusion[:self_top2_hits]} self_top2_rate=#{pct_count(suite_fusion[:self_top2_hits], suite_steps).round(2)} union_k2_hits=#{suite_fusion[:union_k2_hits]} union_k2_rate=#{pct_count(suite_fusion[:union_k2_hits], suite_steps).round(2)} agreement=#{suite_fusion[:agreement]} agreement_hits=#{suite_fusion[:agreement_hits]} agreement_false=#{suite_fusion[:agreement_false]} self_chain_ms=#{suite_fusion[:self_chain_ms].round(3)} mtp_ms=#{suite_fusion[:mtp_ms].round(3)} exact_ms=#{suite_fusion[:self_exact_ms].round(3)}"
        puts "mtp_self_draft_route_policy_suite name=#{suite_prompt[:name]} policy=self_top2_only layers=#{simulate_logit_layers.join(',')} rank=#{rank}#{suite_updown_note} steps=#{suite_steps} resolved_hits=#{suite_fusion[:self_top2_hits]} resolved_rate=#{pct_count(suite_fusion[:self_top2_hits], suite_steps).round(2)} unresolved=#{suite_self_top2_unresolved} attempts=#{suite_self_top2_attempts_total} avg_attempts=#{(suite_self_top2_attempts_total.to_f64 / suite_steps).round(3)} mtp_calls=0 modeled_mtp_ms=0.0"
        puts "mtp_self_draft_route_policy_suite name=#{suite_prompt[:name]} policy=self_top2_first_mtp_k2_on_miss layers=#{simulate_logit_layers.join(',')} rank=#{rank}#{suite_updown_note} steps=#{suite_steps} mtp_width=#{suite_mtp_k2_width} resolved_hits=#{suite_fusion[:union_k2_hits]} resolved_rate=#{pct_count(suite_fusion[:union_k2_hits], suite_steps).round(2)} unresolved=#{suite_steps - suite_fusion[:union_k2_hits]} mtp_calls=#{suite_self_top2_unresolved} mtp_call_rate=#{pct_count(suite_self_top2_unresolved, suite_steps).round(2)} mtp_rescue_hits=#{suite_self_top2_mtp_k2_rescues} mtp_rescue_rate=#{pct_count(suite_self_top2_mtp_k2_rescues, suite_self_top2_unresolved).round(2)} attempts=#{suite_self_top2_mtp_k2_attempts_total} avg_attempts=#{(suite_self_top2_mtp_k2_attempts_total.to_f64 / suite_steps).round(3)} modeled_mtp_ms=#{suite_self_top2_mismatch_mtp_ms.round(3)} saved_mtp_ms=#{suite_self_top2_mismatch_saved_ms.round(3)} note=oracle_policy_pressure_exact_positions_not_resyncing_wall_runtime"
      end
    end
    draft_no_ffn_layer_set = simulate_self_spec_gpu_pipeline_draft_no_ffn_layers.empty? ? nil : Set(Int32).new(simulate_self_spec_gpu_pipeline_draft_no_ffn_layers)
    draft_updown_layer_set = simulate_self_spec_gpu_pipeline_draft_updown_layers.empty? ? nil : Set(Int32).new(simulate_self_spec_gpu_pipeline_draft_updown_layers)
    hybrid_routes = simulate_self_spec_gpu_pipeline_hybrid_sweep ? build_self_spec_hybrid_routes(simulate_logit_layers, draft_no_ffn_layer_set, draft_updown_layer_set, simulate_self_spec_gpu_pipeline_hybrid_rich_sweep) : [] of HybridRoute
    pipeline_gammas = simulate_self_spec_gpu_pipeline_gammas.dup
    if simulate_self_spec_gpu_pipeline > 0 && !pipeline_gammas.includes?(simulate_self_spec_gpu_pipeline)
      pipeline_gammas << simulate_self_spec_gpu_pipeline
    end
    pipeline_route_active = simulate_generate_tokens > 0 && (!pipeline_gammas.empty? || !simulate_self_spec_gpu_pipeline_schedules.empty?)
    if pipeline_route_active
      route_selector_enabled = !simulate_self_spec_gpu_pipeline_route_selector_route.nil? ||
                               !simulate_self_spec_gpu_pipeline_route_selector_no_ffn_layers.empty? ||
                               !simulate_self_spec_gpu_pipeline_route_selector_feature.nil? ||
                               !simulate_self_spec_gpu_pipeline_route_selector_threshold.nil?
      residual_router_enabled = !simulate_self_spec_gpu_pipeline_residual_router_mean_max.nil? ||
                                !simulate_self_spec_gpu_pipeline_residual_router_pass_threshold.nil? ||
                                !simulate_self_spec_gpu_pipeline_residual_router_pass_rate_min.nil?
      value_router_enabled = !simulate_self_spec_gpu_pipeline_value_repeat_rate_min.nil? ||
                             !simulate_self_spec_gpu_pipeline_value_bigram_repeat_rate_min.nil? ||
                             !simulate_self_spec_gpu_pipeline_value_unique_rate_max.nil?
      pre_submit_router_enabled = residual_router_enabled || value_router_enabled
      if pre_submit_router_enabled
        raise "pre-submit routers currently support fixed-gamma pipeline rows; disable --simulate-self-spec-gpu-pipeline-schedule" unless simulate_self_spec_gpu_pipeline_schedules.empty?
        raise "pre-submit routers are not wired into hybrid route scoreboards; disable hybrid sweep" if simulate_self_spec_gpu_pipeline_hybrid_sweep || simulate_self_spec_gpu_pipeline_suite_hybrid_sweep
      end
      selected_route_candidate = nil.as(HybridRoute?)
      pure_route_candidate = nil.as(HybridRoute?)
      if route_selector_enabled
        raise "route selector requires --simulate-self-spec-gpu-pipeline-route-selector-route or --simulate-self-spec-gpu-pipeline-route-selector-no-ffn-layers" if simulate_self_spec_gpu_pipeline_route_selector_route.nil? && simulate_self_spec_gpu_pipeline_route_selector_no_ffn_layers.empty?
        raise "route selector requires --simulate-self-spec-gpu-pipeline-route-selector-feature" if simulate_self_spec_gpu_pipeline_route_selector_feature.nil?
        raise "route selector requires --simulate-self-spec-gpu-pipeline-route-selector-threshold" if simulate_self_spec_gpu_pipeline_route_selector_threshold.nil?
        raise "route selector op must be <= or >=" unless ["<=", ">="].includes?(simulate_self_spec_gpu_pipeline_route_selector_op)
        route_selector_features = ["residual_mean", "residual_p90", "residual_max", "repeat_rate", "bigram_repeat_rate", "unique_rate"]
        raise "route selector feature must be one of #{route_selector_features.join(',')}" unless route_selector_features.includes?(simulate_self_spec_gpu_pipeline_route_selector_feature.not_nil!)
        raise "route selector currently supports fixed-gamma rows; disable --simulate-self-spec-gpu-pipeline-schedule" unless simulate_self_spec_gpu_pipeline_schedules.empty?
        selector_routes = build_self_spec_hybrid_routes(simulate_logit_layers, draft_no_ffn_layer_set, draft_updown_layer_set, simulate_self_spec_gpu_pipeline_hybrid_rich_sweep)
        pure_route_candidate = selector_routes.find { |route| route[:name] == "pure" }
        if simulate_self_spec_gpu_pipeline_route_selector_no_ffn_layers.empty?
          selected_route_candidate = selector_routes.find { |route| route[:name] == simulate_self_spec_gpu_pipeline_route_selector_route }
        else
          custom_noffn = simulate_self_spec_gpu_pipeline_route_selector_no_ffn_layers.uniq.sort
          custom_name = (simulate_self_spec_gpu_pipeline_route_selector_route || "selector_noffn_#{custom_noffn.join('_')}").to_s
          selected_route_candidate = {name: custom_name.gsub(/[^A-Za-z0-9_.-]/, "_"), noffn: Set(Int32).new(custom_noffn).as(Set(Int32)?), updown: nil.as(Set(Int32)?)}
        end
        raise "route selector could not resolve pure route" if pure_route_candidate.nil?
        raise "route selector could not resolve route #{simulate_self_spec_gpu_pipeline_route_selector_route}; run a hybrid sweep to inspect route names" if selected_route_candidate.nil?
        raise "route selector currently supports no-FFN route candidates only" if selected_route_candidate.not_nil![:updown]
      end
      raise "route selector ABBA cycles must be non-negative" if simulate_self_spec_gpu_pipeline_route_selector_abba < 0
      if simulate_self_spec_gpu_pipeline_route_selector_abba > 0
        raise "route selector ABBA requires route selector flags" unless route_selector_enabled
        raise "route selector ABBA is wired only for fixed-gamma non-hybrid rows" if simulate_self_spec_gpu_pipeline_hybrid_sweep || !simulate_self_spec_gpu_pipeline_schedules.empty?
      end
      if residual_router_enabled
        pass_threshold_set = !simulate_self_spec_gpu_pipeline_residual_router_pass_threshold.nil?
        pass_rate_set = !simulate_self_spec_gpu_pipeline_residual_router_pass_rate_min.nil?
        raise "residual router pass threshold and pass-rate min must be set together" if pass_threshold_set != pass_rate_set
      end
      pipeline_mtp_k2_on_reject = if simulate_self_spec_gpu_pipeline_mtp_k2_on_reject
                                    mtp = ML::GGUF::Qwen35MTPWeights.from_safetensors(mtp_path)
                                    mtp.validate_for_qwen35!(weights.hparams)
                                    mtp
                                  else
                                    nil
                                  end
      default_draft_split = ENV["QWEN35_DRAFT_BLOCK_TOKENS"]?.try(&.to_i?) || DEFAULT_SELF_SPEC_GPU_PIPELINE_DRAFT_BLOCK_TOKENS
      pipeline_splits = simulate_self_spec_gpu_pipeline_draft_splits.empty? ? [default_draft_split.as(Int32?)] : simulate_self_spec_gpu_pipeline_draft_splits.map { |v| v.as(Int32?) }
      route_score_rows = [] of RouteScoreRow
      draft_body_score_rows = [] of DraftBodyScoreRow
      risk_offramp_score_rows = [] of RiskOfframpScoreRow
      raise "draft pca-updown repeats must be >= 1" if simulate_self_spec_gpu_pipeline_draft_updown_repeats < 1
      pipeline_updown_base_options = [] of Int32?
      if simulate_self_spec_gpu_pipeline_draft_updown_ranks.empty?
        pipeline_updown_base_options << simulate_self_spec_gpu_pipeline_draft_updown_rank
      else
        simulate_self_spec_gpu_pipeline_draft_updown_ranks.each do |v|
          option = v > 0 ? v : nil
          pipeline_updown_base_options << option unless pipeline_updown_base_options.any? { |existing| existing == option }
        end
      end
      pipeline_updown_options = [] of Int32?
      if simulate_self_spec_gpu_pipeline_draft_updown_repeats <= 1
        pipeline_updown_options.concat(pipeline_updown_base_options)
      else
        ranks = pipeline_updown_base_options.compact
        if ranks.empty? && (single_rank = simulate_self_spec_gpu_pipeline_draft_updown_rank)
          ranks << single_rank
        end
        if ranks.empty?
          pipeline_updown_options.concat(pipeline_updown_base_options)
        else
          simulate_self_spec_gpu_pipeline_draft_updown_repeats.times do |repeat_i|
            if repeat_i.even?
              pipeline_updown_options << nil
              ranks.each { |value| pipeline_updown_options << value }
            else
              ranks.reverse_each { |value| pipeline_updown_options << value }
              pipeline_updown_options << nil
            end
          end
        end
      end
      if !simulate_self_spec_gpu_pipeline_suite_prompts.empty? && pipeline_updown_options.any? { |option| !option.nil? }
        raise "GPU pipeline suite with pca-updown requires external --ffn-pca-calib-prompt so FFN adapters are not tied to the main prompt" if ffn_pca_calib_token_sets.empty?
      end
      if simulate_self_spec_gpu_pipeline_route_selector_abba > 0
        raise "route selector ABBA is not wired with pca-updown draft options" if pipeline_updown_options.any? { |option| !option.nil? }
      end
      state_backup_note = simulate_self_spec_gpu_pipeline_legacy_full_state_backup ? " state_backup=legacy_full" : " state_backup=live_blit"
      exact_refresh_note = ProbeRuntime.gpu_draft_exact_refresh_interval > 0 ? " draft_exact_refresh=#{ProbeRuntime.gpu_draft_exact_refresh_interval}" : ""
      exact_refresh_note += " draft_exact_refresh_prefix=#{ProbeRuntime.gpu_draft_exact_refresh_prefix}" if ProbeRuntime.gpu_draft_exact_refresh_prefix > 0
      exact_refresh_note += " draft_exact_refresh_offsets=#{ProbeRuntime.gpu_draft_exact_refresh_offsets.join(',')}" unless ProbeRuntime.gpu_draft_exact_refresh_offsets.empty?
      exact_refresh_note += " draft_refresh_on_accept=1" if ProbeRuntime.self_spec_draft_refresh_on_accept
      exact_refresh_note += " draft_noffn_fallback=reject" if ProbeRuntime.self_spec_draft_no_ffn_fallback_on_reject
      exact_refresh_note += " draft_noffn_after_full_accepts=#{ProbeRuntime.self_spec_draft_no_ffn_after_full_accepts}" if ProbeRuntime.self_spec_draft_no_ffn_after_full_accepts > 0
      exact_refresh_note += " draft_noffn_min_margin=#{ProbeRuntime.self_spec_draft_no_ffn_min_margin.not_nil!}" if ProbeRuntime.self_spec_draft_no_ffn_min_margin
      exact_refresh_note += " draft_noffn_max_chunks=#{ProbeRuntime.self_spec_draft_no_ffn_max_chunks.not_nil!}" if ProbeRuntime.self_spec_draft_no_ffn_max_chunks
      residual_router_thresholds = self_spec_residual_router_thresholds(thresholds, simulate_self_spec_gpu_pipeline_residual_router_pass_threshold)
      main_route_residual_stats = route_residual_stats(layer_vectors, layer_bases, rank, calib_count, thresholds)
      main_value_stats = self_spec_prompt_value_stats(token_ids, calib_count)
      main_residual_router_note = ""
      main_residual_router_skip = false
      if residual_router_enabled
        main_residual_router_stats = route_residual_stats(layer_vectors, layer_bases, rank, calib_count, residual_router_thresholds)
        main_residual_router_decision = self_spec_residual_router_decision(main_residual_router_stats,
          simulate_self_spec_gpu_pipeline_residual_router_mean_max,
          simulate_self_spec_gpu_pipeline_residual_router_pass_threshold,
          simulate_self_spec_gpu_pipeline_residual_router_pass_rate_min)
        main_residual_router_note = self_spec_residual_router_note(main_residual_router_decision, main_residual_router_stats,
          simulate_self_spec_gpu_pipeline_residual_router_mean_max,
          simulate_self_spec_gpu_pipeline_residual_router_pass_threshold,
          simulate_self_spec_gpu_pipeline_residual_router_pass_rate_min)
        main_residual_router_skip = !main_residual_router_decision[:run]
      end
      main_value_router_note = ""
      main_value_router_skip = false
      if value_router_enabled
        main_value_router_stats = self_spec_prompt_value_stats(token_ids, calib_count)
        main_value_router_decision = self_spec_value_router_decision(main_value_router_stats,
          simulate_self_spec_gpu_pipeline_value_repeat_rate_min,
          simulate_self_spec_gpu_pipeline_value_bigram_repeat_rate_min,
          simulate_self_spec_gpu_pipeline_value_unique_rate_max)
        main_value_router_note = self_spec_value_router_note(main_value_router_decision, main_value_router_stats,
          simulate_self_spec_gpu_pipeline_value_repeat_rate_min,
          simulate_self_spec_gpu_pipeline_value_bigram_repeat_rate_min,
          simulate_self_spec_gpu_pipeline_value_unique_rate_max)
        main_value_router_skip = !main_value_router_decision[:run]
      end
      main_pre_submit_router_note = "#{main_residual_router_note}#{main_value_router_note}"
      main_pre_submit_router_skip = main_residual_router_skip || main_value_router_skip
      route_selector_route_name = route_selector_enabled ? selected_route_candidate.not_nil![:name] : (simulate_self_spec_gpu_pipeline_route_selector_route || "")
      route_selector_feature_name = simulate_self_spec_gpu_pipeline_route_selector_feature || ""
      run_route_selector = ->(scope : String, prompt_name : String, prompt_text : String, prompt_token_ids : Array(Int32), prompt_layer_bases : LayerBasisMap, feature_value : Float64?, force_pure : Bool, abba_index : Int32) {
        if route_selector_enabled
          selector_would_select = route_selector_match?(feature_value, simulate_self_spec_gpu_pipeline_route_selector_op, simulate_self_spec_gpu_pipeline_route_selector_threshold)
          selected = !force_pure && selector_would_select
          route = selected ? selected_route_candidate.not_nil! : pure_route_candidate.not_nil!
          row_prefix = abba_index >= 0 ? "self_spec_gpu_pipeline_route_selector_abba" : "self_spec_gpu_pipeline_route_selector"
          abba_note = abba_index >= 0 ? " abba_index=#{abba_index} mode=#{force_pure ? "pure" : "selector"} route_selector_would_select=#{selector_would_select}" : ""
          pipeline_gammas.each do |pipeline_gamma|
            pipeline_splits.each do |draft_split|
              pipe = simulate_self_spec_gpu_pipeline_run(weights, prompt_token_ids, simulate_generate_tokens, pipeline_gamma, prompt_layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, false, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, nil, nil, ffn_updown_adapters, ffn_updown_adapter_q8_metal, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, simulate_self_spec_gpu_pipeline_draft_updown_min_margin, simulate_self_spec_gpu_pipeline_draft_updown_max_chunks, simulate_self_spec_gpu_pipeline_draft_updown_after_rejects, simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept, simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate, simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps, simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds, !simulate_self_spec_gpu_pipeline_legacy_full_state_backup, route[:noffn], nil, simulate_self_spec_gpu_pipeline_tree2_first, simulate_self_spec_gpu_pipeline_tree2_anywhere, simulate_self_spec_gpu_pipeline_tree2_staged_tokens, simulate_self_spec_gpu_pipeline_tree2_margin_guard, simulate_self_spec_gpu_pipeline_tree2_branch_guard, simulate_self_spec_gpu_pipeline_risk_offramp_margin, pipeline_mtp_k2_on_reject, simulate_self_spec_gpu_pipeline_reject_offramp_after)
              accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
              backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
              split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
              route_note = hybrid_route_note(route, nil)
              selector_note = route_selector_note(route_selector_route_name.to_s, route_selector_feature_name.to_s, simulate_self_spec_gpu_pipeline_route_selector_op, simulate_self_spec_gpu_pipeline_route_selector_threshold, feature_value, selected, route)
              tree2_note = (simulate_self_spec_gpu_pipeline_tree2_first || simulate_self_spec_gpu_pipeline_tree2_anywhere || simulate_self_spec_gpu_pipeline_tree2_staged_tokens > 0 || !simulate_self_spec_gpu_pipeline_tree2_margin_guard.nil? || !simulate_self_spec_gpu_pipeline_tree2_branch_guard.nil? || !simulate_self_spec_gpu_pipeline_risk_offramp_margin.nil? || simulate_self_spec_gpu_pipeline_mtp_k2_on_reject || simulate_self_spec_gpu_pipeline_reject_offramp_after > 0) ? self_spec_pipeline_tree2_note(pipe) : ""
              attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
              puts "#{row_prefix} scope=#{scope} name=#{prompt_name}#{abba_note} layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{pipeline_gamma}#{split_note}#{route_note}#{selector_note}#{exact_refresh_note}#{backup_note}#{state_backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{tree2_note}#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
              if dump_path = simulate_self_spec_gpu_pipeline_dump_cycles_path
                dump_self_spec_gpu_pipeline_cycles(dump_path.not_nil!, prompt_name, prompt_text, "self_lowrank/gamma=#{pipeline_gamma}/route_selector=#{route[:name]}", simulate_logit_layers, rank, "gamma=#{pipeline_gamma}", pipe)
              end
            end
          end
        end
      }
      raise "risk-offramp repeats must be >= 1" if simulate_self_spec_gpu_pipeline_risk_offramp_repeats < 1
      risk_offramp_base_options = [] of Float64?
      add_risk_offramp_option = ->(value : Float64?) {
        risk_offramp_base_options << value unless risk_offramp_base_options.any? { |existing| existing == value }
      }
      if simulate_self_spec_gpu_pipeline_risk_offramp_margins.empty?
        add_risk_offramp_option.call(simulate_self_spec_gpu_pipeline_risk_offramp_margin)
      else
        add_risk_offramp_option.call(nil)
        simulate_self_spec_gpu_pipeline_risk_offramp_margins.each { |value| add_risk_offramp_option.call(value) }
      end
      risk_offramp_options = [] of Float64?
      if simulate_self_spec_gpu_pipeline_risk_offramp_repeats <= 1
        risk_offramp_options.concat(risk_offramp_base_options)
      else
        thresholds = risk_offramp_base_options.compact
        if thresholds.empty? && (single_margin = simulate_self_spec_gpu_pipeline_risk_offramp_margin)
          thresholds << single_margin
        end
        if thresholds.empty?
          risk_offramp_options.concat(risk_offramp_base_options)
        else
          simulate_self_spec_gpu_pipeline_risk_offramp_repeats.times do |repeat_i|
            if repeat_i.even?
              risk_offramp_options << nil
              thresholds.each { |value| risk_offramp_options << value }
            else
              thresholds.reverse_each { |value| risk_offramp_options << value }
              risk_offramp_options << nil
            end
          end
        end
      end
      if simulate_self_spec_gpu_pipeline_hybrid_sweep && !simulate_self_spec_gpu_pipeline_risk_offramp_margins.empty?
        raise "risk-offramp margin sweep is not wired into hybrid route scoreboards yet; use a single --simulate-self-spec-gpu-pipeline-risk-offramp-margin=F or disable hybrid sweep"
      end
      branch_snapshot_base_enabled = ProbeRuntime.self_spec_branch_guard_snapshot
      branch_snapshot_single_pass_base_enabled = ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint
      branch_snapshot_only_split_base_enabled = ProbeRuntime.self_spec_branch_guard_snapshot_only_split
      branch_snapshot_min_prefix_base = ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix
      branch_snapshot_suffix_min_threshold_base = ProbeRuntime.self_spec_branch_guard_snapshot_suffix_min_threshold
      branch_snapshot_suffix_threshold_base = ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold
      branch_snapshot_prefix_suffix_thresholds_base = ProbeRuntime.self_spec_branch_guard_snapshot_prefix_suffix_thresholds
      branch_snapshot_no_snapshot_threshold_base = ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold
      branch_snapshot_mode_options = [] of NamedTuple(name: String, snapshot: Bool, onepass: Bool, only_split: Bool, min_prefix: Int32?, suffix_min_threshold: Float64?, suffix_threshold: Float64?, no_snapshot_threshold: Float64?)
      if simulate_self_spec_gpu_pipeline_branch_snapshot_modes.empty?
        branch_snapshot_policy_name = simulate_self_spec_gpu_pipeline_branch_snapshot_policy || ""
        branch_snapshot_mode_options << {name: branch_snapshot_policy_name.to_s, snapshot: branch_snapshot_base_enabled, onepass: branch_snapshot_single_pass_base_enabled, only_split: branch_snapshot_only_split_base_enabled, min_prefix: nil, suffix_min_threshold: nil, suffix_threshold: nil, no_snapshot_threshold: nil}
      else
        if simulate_self_spec_gpu_pipeline_hybrid_sweep
          raise "branch snapshot mode sweep is not wired into hybrid route scoreboards yet; disable hybrid sweep"
        end
        unless simulate_self_spec_gpu_pipeline_schedules.empty?
          raise "branch snapshot mode sweep is not wired into schedule routes yet; use fixed gammas"
        end
        simulate_self_spec_gpu_pipeline_branch_snapshot_modes.each do |mode|
          case mode
          when "nosnap"
            branch_snapshot_mode_options << {name: "nosnap", snapshot: false, onepass: false, only_split: false, min_prefix: nil, suffix_min_threshold: nil, suffix_threshold: nil, no_snapshot_threshold: nil}
          when "split"
            branch_snapshot_mode_options << {name: "split", snapshot: true, onepass: false, only_split: false, min_prefix: nil, suffix_min_threshold: nil, suffix_threshold: nil, no_snapshot_threshold: nil}
          when "split_suffix2"
            branch_snapshot_mode_options << {name: "split_suffix2", snapshot: true, onepass: false, only_split: true, min_prefix: nil, suffix_min_threshold: nil, suffix_threshold: nil, no_snapshot_threshold: nil}
          when "split_min3_suffix2"
            branch_snapshot_mode_options << {name: "split_min3_suffix2", snapshot: true, onepass: false, only_split: true, min_prefix: 3, suffix_min_threshold: nil, suffix_threshold: 2.0, no_snapshot_threshold: nil}
          when "split_min3_suffix2_keepguard"
            branch_snapshot_mode_options << {name: "split_min3_suffix2_keepguard", snapshot: true, onepass: false, only_split: false, min_prefix: 3, suffix_min_threshold: nil, suffix_threshold: 2.0, no_snapshot_threshold: nil}
          when "split_min3_suffix2_guard02"
            branch_snapshot_mode_options << {name: "split_min3_suffix2_guard02", snapshot: true, onepass: false, only_split: true, min_prefix: 3, suffix_min_threshold: nil, suffix_threshold: 2.0, no_snapshot_threshold: 0.2}
          when "onepass"
            branch_snapshot_mode_options << {name: "onepass", snapshot: true, onepass: true, only_split: false, min_prefix: nil, suffix_min_threshold: nil, suffix_threshold: nil, no_snapshot_threshold: nil}
          when "onepass_suffix2"
            branch_snapshot_mode_options << {name: "onepass_suffix2", snapshot: true, onepass: true, only_split: true, min_prefix: nil, suffix_min_threshold: nil, suffix_threshold: nil, no_snapshot_threshold: nil}
          when "onepass_min3_suffix2"
            branch_snapshot_mode_options << {name: "onepass_min3_suffix2", snapshot: true, onepass: true, only_split: true, min_prefix: 3, suffix_min_threshold: nil, suffix_threshold: 2.0, no_snapshot_threshold: nil}
          when "onepass_min3_suffix2_keepguard"
            branch_snapshot_mode_options << {name: "onepass_min3_suffix2_keepguard", snapshot: true, onepass: true, only_split: false, min_prefix: 3, suffix_min_threshold: nil, suffix_threshold: 2.0, no_snapshot_threshold: nil}
          when "onepass_min3_suffix2_guard02"
            branch_snapshot_mode_options << {name: "onepass_min3_suffix2_guard02", snapshot: true, onepass: true, only_split: true, min_prefix: 3, suffix_min_threshold: nil, suffix_threshold: 2.0, no_snapshot_threshold: 0.2}
          when "onepass_min3_suffix2_guard005"
            branch_snapshot_mode_options << {name: "onepass_min3_suffix2_guard005", snapshot: true, onepass: true, only_split: true, min_prefix: 3, suffix_min_threshold: nil, suffix_threshold: 2.0, no_snapshot_threshold: 0.05}
          when "onepass_min3_suffix2_guard01"
            branch_snapshot_mode_options << {name: "onepass_min3_suffix2_guard01", snapshot: true, onepass: true, only_split: true, min_prefix: 3, suffix_min_threshold: nil, suffix_threshold: 2.0, no_snapshot_threshold: 0.1}
          when "onepass_min3_suffix1to2_guard01"
            branch_snapshot_mode_options << {name: "onepass_min3_suffix1to2_guard01", snapshot: true, onepass: true, only_split: true, min_prefix: 3, suffix_min_threshold: 1.0, suffix_threshold: 2.0, no_snapshot_threshold: 0.1}
          when "onepass_min3_suffix2_guard05"
            branch_snapshot_mode_options << {name: "onepass_min3_suffix2_guard05", snapshot: true, onepass: true, only_split: true, min_prefix: 3, suffix_min_threshold: nil, suffix_threshold: 2.0, no_snapshot_threshold: 0.5}
          when "onepass_min3_suffix2_guardinf"
            branch_snapshot_mode_options << {name: "onepass_min3_suffix2_guardinf", snapshot: true, onepass: true, only_split: true, min_prefix: 3, suffix_min_threshold: nil, suffix_threshold: 2.0, no_snapshot_threshold: Float64::INFINITY}
          else
            raise "unknown branch snapshot mode #{mode.inspect}; expected nosnap, split, split_suffix2, split_min3_suffix2, split_min3_suffix2_keepguard, split_min3_suffix2_guard02, onepass, onepass_suffix2, onepass_min3_suffix2, onepass_min3_suffix2_keepguard, onepass_min3_suffix1to2_guard01, or onepass_min3_suffix2_guard005/_guard01/_guard02/_guard05/_guardinf"
          end
        end
      end
      apply_branch_snapshot_mode = ->(branch_snapshot_mode : NamedTuple(name: String, snapshot: Bool, onepass: Bool, only_split: Bool, min_prefix: Int32?, suffix_min_threshold: Float64?, suffix_threshold: Float64?, no_snapshot_threshold: Float64?)) {
        ProbeRuntime.self_spec_branch_guard_snapshot = branch_snapshot_mode[:snapshot]
        ProbeRuntime.self_spec_branch_guard_single_pass_checkpoint = branch_snapshot_mode[:onepass]
        ProbeRuntime.self_spec_branch_guard_snapshot_only_split = branch_snapshot_mode[:only_split]
        ProbeRuntime.self_spec_branch_guard_snapshot_min_prefix = branch_snapshot_mode[:min_prefix] || branch_snapshot_min_prefix_base
        ProbeRuntime.self_spec_branch_guard_no_snapshot_threshold = branch_snapshot_mode[:no_snapshot_threshold] || branch_snapshot_no_snapshot_threshold_base
        if suffix_threshold = branch_snapshot_mode[:suffix_threshold]
          ProbeRuntime.self_spec_branch_guard_snapshot_suffix_min_threshold = branch_snapshot_mode[:suffix_min_threshold]
          ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = suffix_threshold
          ProbeRuntime.self_spec_branch_guard_snapshot_prefix_suffix_thresholds = [] of Tuple(Int32, Float64)
        else
          ProbeRuntime.self_spec_branch_guard_snapshot_suffix_min_threshold = branch_snapshot_suffix_min_threshold_base
          ProbeRuntime.self_spec_branch_guard_snapshot_suffix_threshold = branch_snapshot_suffix_threshold_base
          ProbeRuntime.self_spec_branch_guard_snapshot_prefix_suffix_thresholds = branch_snapshot_prefix_suffix_thresholds_base
        end
      }
      if simulate_self_spec_gpu_pipeline_hybrid_sweep && simulate_self_spec_gpu_pipeline_draft_updown_repeats > 1
        raise "draft pca-updown repeats are not wired into hybrid route scoreboards yet; disable hybrid sweep or set repeats=1"
      end
      raise "no-FFN fallback ABBA cycles must be non-negative" if simulate_self_spec_gpu_pipeline_draft_no_ffn_fallback_abba < 0
      noffn_fallback_abba_modes = [] of Bool
      if simulate_self_spec_gpu_pipeline_draft_no_ffn_fallback_abba > 0
        raise "no-FFN fallback ABBA requires a no-FFN draft route" unless simulate_self_spec_gpu_pipeline_draft_no_ffn || draft_no_ffn_layer_set
        raise "no-FFN fallback ABBA is wired only for fixed-gamma non-hybrid rows" if simulate_self_spec_gpu_pipeline_hybrid_sweep || !simulate_self_spec_gpu_pipeline_schedules.empty?
        raise "no-FFN fallback ABBA is not wired with pca-updown draft options" if pipeline_updown_options.any? { |option| !option.nil? }
        raise "no-FFN fallback ABBA is not wired with risk-offramp sweeps" if risk_offramp_options.any? { |option| !option.nil? }
        raise "no-FFN fallback ABBA is not wired with branch snapshot sweeps" if branch_snapshot_mode_options.any? { |mode| !mode[:name].empty? }
        simulate_self_spec_gpu_pipeline_draft_no_ffn_fallback_abba.times do
          noffn_fallback_abba_modes.concat([false, true, true, false])
        end
      end
      noffn_fallback_base_enabled = ProbeRuntime.self_spec_draft_no_ffn_fallback_on_reject
      run_noffn_fallback_abba = ->(scope : String, prompt_name : String, prompt_token_ids : Array(Int32), prompt_layer_bases : LayerBasisMap) {
        noffn_fallback_abba_modes.each_with_index do |fallback_enabled, abba_index|
          ProbeRuntime.self_spec_draft_no_ffn_fallback_on_reject = fallback_enabled
          pipeline_gammas.each do |pipeline_gamma|
            pipeline_splits.each do |draft_split|
              pipe = simulate_self_spec_gpu_pipeline_run(weights, prompt_token_ids, simulate_generate_tokens, pipeline_gamma, prompt_layer_bases, rank,
                !simulate_self_spec_gpu_pipeline_no_backup, draft_split, simulate_self_spec_gpu_pipeline_draft_no_ffn,
                simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, nil, nil, ffn_updown_adapters,
                ffn_updown_adapter_q8_metal,
                simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts,
                simulate_self_spec_gpu_pipeline_draft_updown_min_margin, simulate_self_spec_gpu_pipeline_draft_updown_max_chunks,
                simulate_self_spec_gpu_pipeline_draft_updown_after_rejects, simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept,
                simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate,
                simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps, simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds,
                !simulate_self_spec_gpu_pipeline_legacy_full_state_backup, draft_no_ffn_layer_set, draft_updown_layer_set,
                simulate_self_spec_gpu_pipeline_tree2_first, simulate_self_spec_gpu_pipeline_tree2_anywhere,
                simulate_self_spec_gpu_pipeline_tree2_staged_tokens, simulate_self_spec_gpu_pipeline_tree2_margin_guard,
                simulate_self_spec_gpu_pipeline_tree2_branch_guard, nil, pipeline_mtp_k2_on_reject,
                simulate_self_spec_gpu_pipeline_reject_offramp_after)
              accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
              split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
              fallback_mode = fallback_enabled ? "on" : "off"
              fallback_note = fallback_enabled ? " draft_noffn_fallback=reject" : ""
              draft_variant_note = simulate_self_spec_gpu_pipeline_draft_no_ffn ? " draft_no_ffn=1" : ""
              draft_no_ffn_layers_note = draft_no_ffn_layer_set ? " draft_no_ffn_layers=#{draft_no_ffn_layer_set.not_nil!.to_a.sort.join(',')}" : ""
              puts "self_spec_gpu_pipeline_noffn_fallback_abba scope=#{scope} name=#{prompt_name} abba_index=#{abba_index} mode=#{fallback_mode} layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{pipeline_gamma}#{split_note}#{draft_variant_note}#{draft_no_ffn_layers_note}#{fallback_note}#{state_backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
            end
          end
        end
        ProbeRuntime.self_spec_draft_no_ffn_fallback_on_reject = noffn_fallback_base_enabled
      }
      route_selector_abba_modes = [] of Bool
      simulate_self_spec_gpu_pipeline_route_selector_abba.times do
        route_selector_abba_modes.concat([true, false, false, true])
      end
      run_route_selector_abba = ->(scope : String, prompt_name : String, prompt_text : String, prompt_token_ids : Array(Int32), prompt_layer_bases : LayerBasisMap, feature_value : Float64?) {
        route_selector_abba_modes.each_with_index do |force_pure, abba_index|
          run_route_selector.call(scope, prompt_name, prompt_text, prompt_token_ids, prompt_layer_bases, feature_value, force_pure, abba_index)
        end
      }
      ProbeRuntime.self_spec_router_trace_label = main_prompt_name
      run_noffn_fallback_abba.call("main", main_prompt_name, token_ids, layer_bases)
      main_route_selector_value = prompt_route_selector_feature_value(main_route_residual_stats, main_value_stats, route_selector_feature_name.to_s)
      if simulate_self_spec_gpu_pipeline_route_selector_abba > 0
        run_route_selector_abba.call("main", main_prompt_name, prompt, token_ids, layer_bases, main_route_selector_value)
      else
        run_route_selector.call("main", main_prompt_name, prompt, token_ids, layer_bases, main_route_selector_value, false, -1)
      end
      if simulate_self_spec_gpu_pipeline_hybrid_sweep
        pipeline_gammas.each do |pipeline_gamma|
          pipeline_splits.each do |draft_split|
            pipeline_updown_options.each do |pipeline_updown_rank|
              hybrid_routes.each do |route|
                next if pipeline_updown_rank && route[:updown].nil?
                next if pipeline_updown_rank.nil? && route[:updown]
                route_updown_rank = route[:updown] ? pipeline_updown_rank : nil
                pipe = simulate_self_spec_gpu_pipeline_run(weights, token_ids, simulate_generate_tokens, pipeline_gamma, layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, false, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, nil, route_updown_rank, ffn_updown_adapters, ffn_updown_adapter_q8_metal, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, simulate_self_spec_gpu_pipeline_draft_updown_min_margin, simulate_self_spec_gpu_pipeline_draft_updown_max_chunks, simulate_self_spec_gpu_pipeline_draft_updown_after_rejects, simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept, simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate, simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps, simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds, !simulate_self_spec_gpu_pipeline_legacy_full_state_backup, route[:noffn], route[:updown], simulate_self_spec_gpu_pipeline_tree2_first, simulate_self_spec_gpu_pipeline_tree2_anywhere, simulate_self_spec_gpu_pipeline_tree2_staged_tokens, simulate_self_spec_gpu_pipeline_tree2_margin_guard, simulate_self_spec_gpu_pipeline_tree2_branch_guard, simulate_self_spec_gpu_pipeline_risk_offramp_margin, pipeline_mtp_k2_on_reject, simulate_self_spec_gpu_pipeline_reject_offramp_after)
                accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                draft_updown_note = route_updown_rank ? " draft_pca_updown=#{route_updown_rank}" : ""
                draft_updown_fallback_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                draft_updown_warmup_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                draft_updown_margin_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_min_margin) ? " draft_pca_updown_min_margin=#{simulate_self_spec_gpu_pipeline_draft_updown_min_margin}" : ""
                split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                route_note = hybrid_route_note(route, route_updown_rank)
                tree2_note = (simulate_self_spec_gpu_pipeline_tree2_first || simulate_self_spec_gpu_pipeline_tree2_anywhere || simulate_self_spec_gpu_pipeline_tree2_staged_tokens > 0 || !simulate_self_spec_gpu_pipeline_tree2_margin_guard.nil? || !simulate_self_spec_gpu_pipeline_tree2_branch_guard.nil? || !simulate_self_spec_gpu_pipeline_risk_offramp_margin.nil? || !simulate_self_spec_gpu_pipeline_draft_updown_min_margin.nil? || simulate_self_spec_gpu_pipeline_mtp_k2_on_reject || simulate_self_spec_gpu_pipeline_reject_offramp_after > 0) ? self_spec_pipeline_tree2_note(pipe) : ""
                attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                agreement_note = simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate ? self_spec_pipeline_updown_agreement_note(pipe) : ""
                puts "self_spec_gpu_pipeline_hybrid layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{pipeline_gamma}#{split_note}#{route_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{draft_updown_margin_note}#{exact_refresh_note}#{backup_note}#{state_backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} draft_noffn_chunks=#{pipe[:draft_noffn_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{tree2_note}#{agreement_note}#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
                if dump_path = simulate_self_spec_gpu_pipeline_dump_cycles_path
                  route_label = route_updown_rank ? "#{route[:name]}_updown#{route_updown_rank}" : route[:name]
                  dump_self_spec_gpu_pipeline_cycles(dump_path.not_nil!, main_prompt_name, prompt, "self_lowrank/gamma=#{pipeline_gamma}/route=#{route_label}", simulate_logit_layers, rank, "gamma=#{pipeline_gamma}", pipe)
                end
                append_route_score(route_score_rows, main_prompt_name, "gamma=#{pipeline_gamma}", route, draft_split, route_updown_rank, pipe, accept_rate,
                  main_route_residual_stats[:mean], main_route_residual_stats[:p90], main_route_residual_stats[:max],
                  main_value_stats[:repeat_rate], main_value_stats[:bigram_repeat_rate], main_value_stats[:unique_rate])
              end
            end
          end
        end
        simulate_self_spec_gpu_pipeline_schedules.each do |pipeline_schedule|
          next if pipeline_schedule.empty?
          pipeline_splits.each do |draft_split|
            pipeline_updown_options.each do |pipeline_updown_rank|
              hybrid_routes.each do |route|
                next if pipeline_updown_rank && route[:updown].nil?
                next if pipeline_updown_rank.nil? && route[:updown]
                route_updown_rank = route[:updown] ? pipeline_updown_rank : nil
                pipe = simulate_self_spec_gpu_pipeline_run(weights, token_ids, simulate_generate_tokens, pipeline_schedule[0], layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, false, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, pipeline_schedule, route_updown_rank, ffn_updown_adapters, ffn_updown_adapter_q8_metal, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, simulate_self_spec_gpu_pipeline_draft_updown_min_margin, simulate_self_spec_gpu_pipeline_draft_updown_max_chunks, simulate_self_spec_gpu_pipeline_draft_updown_after_rejects, simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept, simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate, simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps, simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds, !simulate_self_spec_gpu_pipeline_legacy_full_state_backup, route[:noffn], route[:updown], simulate_self_spec_gpu_pipeline_tree2_first, simulate_self_spec_gpu_pipeline_tree2_anywhere, simulate_self_spec_gpu_pipeline_tree2_staged_tokens, simulate_self_spec_gpu_pipeline_tree2_margin_guard, simulate_self_spec_gpu_pipeline_tree2_branch_guard, simulate_self_spec_gpu_pipeline_risk_offramp_margin, pipeline_mtp_k2_on_reject, simulate_self_spec_gpu_pipeline_reject_offramp_after)
                accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                draft_updown_note = route_updown_rank ? " draft_pca_updown=#{route_updown_rank}" : ""
                draft_updown_fallback_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                draft_updown_warmup_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                draft_updown_margin_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_min_margin) ? " draft_pca_updown_min_margin=#{simulate_self_spec_gpu_pipeline_draft_updown_min_margin}" : ""
                split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                route_note = hybrid_route_note(route, route_updown_rank)
                tree2_note = (simulate_self_spec_gpu_pipeline_tree2_first || simulate_self_spec_gpu_pipeline_tree2_anywhere || simulate_self_spec_gpu_pipeline_tree2_staged_tokens > 0 || !simulate_self_spec_gpu_pipeline_tree2_margin_guard.nil? || !simulate_self_spec_gpu_pipeline_tree2_branch_guard.nil? || !simulate_self_spec_gpu_pipeline_risk_offramp_margin.nil? || !simulate_self_spec_gpu_pipeline_draft_updown_min_margin.nil? || simulate_self_spec_gpu_pipeline_mtp_k2_on_reject || simulate_self_spec_gpu_pipeline_reject_offramp_after > 0) ? self_spec_pipeline_tree2_note(pipe) : ""
                attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                agreement_note = simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate ? self_spec_pipeline_updown_agreement_note(pipe) : ""
                puts "self_spec_gpu_pipeline_hybrid layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{pipeline_schedule.join(',')}#{split_note}#{route_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{draft_updown_margin_note}#{exact_refresh_note}#{backup_note}#{state_backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} draft_noffn_chunks=#{pipe[:draft_noffn_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{tree2_note}#{agreement_note}#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
                if dump_path = simulate_self_spec_gpu_pipeline_dump_cycles_path
                  route_label = route_updown_rank ? "#{route[:name]}_updown#{route_updown_rank}" : route[:name]
                  schedule_label = "schedule=#{pipeline_schedule.join(',')}"
                  dump_self_spec_gpu_pipeline_cycles(dump_path.not_nil!, main_prompt_name, prompt, "self_lowrank/#{schedule_label}/route=#{route_label}", simulate_logit_layers, rank, schedule_label, pipe)
                end
                append_route_score(route_score_rows, main_prompt_name, "schedule=#{pipeline_schedule.join(',')}", route, draft_split, route_updown_rank, pipe, accept_rate,
                  main_route_residual_stats[:mean], main_route_residual_stats[:p90], main_route_residual_stats[:max],
                  main_value_stats[:repeat_rate], main_value_stats[:bigram_repeat_rate], main_value_stats[:unique_rate])
              end
            end
          end
        end
      else
        pipeline_gammas.each do |pipeline_gamma|
          pipeline_splits.each do |draft_split|
            pipeline_updown_options.each do |pipeline_updown_rank|
              risk_offramp_options.each do |risk_offramp_margin|
                branch_snapshot_mode_options.each do |branch_snapshot_mode|
                  apply_branch_snapshot_mode.call(branch_snapshot_mode)
                  effective_pipeline_updown_rank = prompt_category_allowed?(main_prompt_name, simulate_self_spec_gpu_pipeline_draft_updown_categories) ? pipeline_updown_rank : nil
                  if main_pre_submit_router_skip
                    backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                    draft_variant_note = simulate_self_spec_gpu_pipeline_draft_no_ffn ? " draft_no_ffn=1" : ""
                    draft_no_ffn_layers_note = draft_no_ffn_layer_set ? " draft_no_ffn_layers=#{draft_no_ffn_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                    draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                    draft_updown_note = effective_pipeline_updown_rank ? " draft_pca_updown=#{effective_pipeline_updown_rank}" : ""
                    draft_updown_category_note = (pipeline_updown_rank && effective_pipeline_updown_rank.nil? && !simulate_self_spec_gpu_pipeline_draft_updown_categories.empty?) ? " draft_pca_updown_category_skip=#{probe_prompt_category(main_prompt_name)}" : ""
                    draft_updown_layers_note = (effective_pipeline_updown_rank && draft_updown_layer_set) ? " draft_pca_updown_layers=#{draft_updown_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                    draft_updown_fallback_note = (effective_pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                    draft_updown_warmup_note = (effective_pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                    draft_updown_margin_note = (effective_pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_min_margin) ? " draft_pca_updown_min_margin=#{simulate_self_spec_gpu_pipeline_draft_updown_min_margin}" : ""
                    split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                    risk_offramp_note = risk_offramp_margin ? " risk_offramp_margin=#{risk_offramp_margin}" : ""
                    branch_snapshot_mode_note = branch_snapshot_mode[:name].empty? ? "" : " branch_snapshot_mode=#{branch_snapshot_mode[:name]}"
                    exact = simulate_self_spec_gpu_pipeline_exact_fallback_run(weights, token_ids, simulate_generate_tokens)
                    puts "self_spec_gpu_pipeline layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{pipeline_gamma}#{branch_snapshot_mode_note}#{split_note}#{draft_variant_note}#{draft_no_ffn_layers_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_category_note}#{draft_updown_layers_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{draft_updown_margin_note}#{risk_offramp_note}#{main_pre_submit_router_note}#{exact_refresh_note}#{backup_note}#{state_backup_note}#{self_spec_pipeline_exact_fallback_fields(exact, simulate_generate_tokens)}"
                    next
                  end
                  pipe = simulate_self_spec_gpu_pipeline_run(weights, token_ids, simulate_generate_tokens, pipeline_gamma, layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, simulate_self_spec_gpu_pipeline_draft_no_ffn, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, nil, effective_pipeline_updown_rank, ffn_updown_adapters, ffn_updown_adapter_q8_metal, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, simulate_self_spec_gpu_pipeline_draft_updown_min_margin, simulate_self_spec_gpu_pipeline_draft_updown_max_chunks, simulate_self_spec_gpu_pipeline_draft_updown_after_rejects, simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept, simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate, simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps, simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds, !simulate_self_spec_gpu_pipeline_legacy_full_state_backup, draft_no_ffn_layer_set, draft_updown_layer_set, simulate_self_spec_gpu_pipeline_tree2_first, simulate_self_spec_gpu_pipeline_tree2_anywhere, simulate_self_spec_gpu_pipeline_tree2_staged_tokens, simulate_self_spec_gpu_pipeline_tree2_margin_guard, simulate_self_spec_gpu_pipeline_tree2_branch_guard, risk_offramp_margin, pipeline_mtp_k2_on_reject, simulate_self_spec_gpu_pipeline_reject_offramp_after)
                  accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                  backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                  draft_variant_note = simulate_self_spec_gpu_pipeline_draft_no_ffn ? " draft_no_ffn=1" : ""
                  draft_no_ffn_layers_note = draft_no_ffn_layer_set ? " draft_no_ffn_layers=#{draft_no_ffn_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                  draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                  draft_updown_note = effective_pipeline_updown_rank ? " draft_pca_updown=#{effective_pipeline_updown_rank}" : ""
                  draft_updown_category_note = (pipeline_updown_rank && effective_pipeline_updown_rank.nil? && !simulate_self_spec_gpu_pipeline_draft_updown_categories.empty?) ? " draft_pca_updown_category_skip=#{probe_prompt_category(main_prompt_name)}" : ""
                  draft_updown_layers_note = (effective_pipeline_updown_rank && draft_updown_layer_set) ? " draft_pca_updown_layers=#{draft_updown_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                  draft_updown_fallback_note = (effective_pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                  draft_updown_warmup_note = (effective_pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                  draft_updown_margin_note = (effective_pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_min_margin) ? " draft_pca_updown_min_margin=#{simulate_self_spec_gpu_pipeline_draft_updown_min_margin}" : ""
                  split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                  risk_offramp_note = risk_offramp_margin ? " risk_offramp_margin=#{risk_offramp_margin}" : ""
                  branch_snapshot_mode_note = branch_snapshot_mode[:name].empty? ? "" : " branch_snapshot_mode=#{branch_snapshot_mode[:name]}"
                  branch_snapshot_mode_score_note = branch_snapshot_mode[:name].empty? ? "" : "/branch_snapshot=#{branch_snapshot_mode[:name]}"
                  tree2_note = (simulate_self_spec_gpu_pipeline_tree2_first || simulate_self_spec_gpu_pipeline_tree2_anywhere || simulate_self_spec_gpu_pipeline_tree2_staged_tokens > 0 || !simulate_self_spec_gpu_pipeline_tree2_margin_guard.nil? || !simulate_self_spec_gpu_pipeline_tree2_branch_guard.nil? || !risk_offramp_margin.nil? || !simulate_self_spec_gpu_pipeline_draft_updown_min_margin.nil? || simulate_self_spec_gpu_pipeline_mtp_k2_on_reject || simulate_self_spec_gpu_pipeline_reject_offramp_after > 0) ? self_spec_pipeline_tree2_note(pipe) : ""
                  attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                  agreement_note = simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate ? self_spec_pipeline_updown_agreement_note(pipe) : ""
                  puts "self_spec_gpu_pipeline layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{pipeline_gamma}#{branch_snapshot_mode_note}#{split_note}#{draft_variant_note}#{draft_no_ffn_layers_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_category_note}#{draft_updown_layers_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{draft_updown_margin_note}#{risk_offramp_note}#{main_pre_submit_router_note}#{exact_refresh_note}#{backup_note}#{state_backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} draft_noffn_chunks=#{pipe[:draft_noffn_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{tree2_note}#{agreement_note}#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
                  if !route_memory_learned &&
                     route_memory_entry.nil? &&
                     (store = route_memory_store) &&
                     (threshold = simulate_self_spec_gpu_pipeline_draft_updown_first_margin_threshold) &&
                     pipeline_updown_rank &&
                     risk_offramp_margin.nil? &&
                     branch_snapshot_mode[:name].empty?
                    learned_route = pipe[:draft_updown_chunks] > 0 && effective_pipeline_updown_rank ? ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN : ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_BASELINE
                    learned_rank = learned_route == ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN ? effective_pipeline_updown_rank : nil
                    learned_layers = learned_route == ML::GGUF::Qwen35PromptCache::PROPOSAL_ROUTE_PCA_UPDOWN && draft_updown_layer_set ? draft_updown_layer_set.not_nil!.to_a.sort : [] of Int32
                    evidence = "gamma=#{pipeline_gamma} split=#{draft_split || "default"} accept=#{accept_rate.round(2)} overlap_ms=#{pipe[:overlap_ms].round(3)} updown_chunks=#{pipe[:draft_updown_chunks]} parity=#{pipe[:parity]}"
                    saved_route = store.save_proposal_route(
                      model_id: route_memory_model_id.not_nil!,
                      tokenizer_id: route_memory_tokenizer_id.not_nil!,
                      prompt_text: prompt,
                      token_ids: token_ids,
                      route: learned_route,
                      route_rank: learned_rank,
                      route_layers: learned_layers,
                      route_key: simulate_self_spec_gpu_pipeline_draft_updown_route_key,
                      trigger: "first-margin<=#{threshold}",
                      evidence: evidence,
                    )
                    route_memory_learned = true
                    saved_rank = saved_route.route_rank ? saved_route.route_rank.to_s : "na"
                    saved_layers = saved_route.route_layers.empty? ? "default" : saved_route.route_layers.join(",")
                    puts "proposal_route_memory_saved route=#{saved_route.route} rank=#{saved_rank} layers=#{saved_layers} key=#{simulate_self_spec_gpu_pipeline_draft_updown_route_key || "exact_prompt"}"
                  end
                  if dump_path = simulate_self_spec_gpu_pipeline_dump_cycles_path
                    dump_self_spec_gpu_pipeline_cycles(dump_path.not_nil!, main_prompt_name, prompt, "self_lowrank/gamma=#{pipeline_gamma}", simulate_logit_layers, rank, "gamma=#{pipeline_gamma}", pipe)
                  end
                  append_draft_body_score(draft_body_score_rows, main_prompt_name, "gamma=#{pipeline_gamma}#{branch_snapshot_mode_score_note}", draft_split, effective_pipeline_updown_rank, pipe, accept_rate)
                  append_risk_offramp_score(risk_offramp_score_rows, main_prompt_name, "gamma=#{pipeline_gamma}#{branch_snapshot_mode_score_note}", draft_split, risk_offramp_margin, pipe, accept_rate)
                end
              end
            end
          end
        end
        simulate_self_spec_gpu_pipeline_schedules.each do |pipeline_schedule|
          next if pipeline_schedule.empty?
          pipeline_splits.each do |draft_split|
            pipeline_updown_options.each do |pipeline_updown_rank|
              risk_offramp_options.each do |risk_offramp_margin|
                pipe = simulate_self_spec_gpu_pipeline_run(weights, token_ids, simulate_generate_tokens, pipeline_schedule[0], layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, simulate_self_spec_gpu_pipeline_draft_no_ffn, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, pipeline_schedule, pipeline_updown_rank, ffn_updown_adapters, ffn_updown_adapter_q8_metal, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, simulate_self_spec_gpu_pipeline_draft_updown_min_margin, simulate_self_spec_gpu_pipeline_draft_updown_max_chunks, simulate_self_spec_gpu_pipeline_draft_updown_after_rejects, simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept, simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate, simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps, simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds, !simulate_self_spec_gpu_pipeline_legacy_full_state_backup, draft_no_ffn_layer_set, draft_updown_layer_set, simulate_self_spec_gpu_pipeline_tree2_first, simulate_self_spec_gpu_pipeline_tree2_anywhere, simulate_self_spec_gpu_pipeline_tree2_staged_tokens, simulate_self_spec_gpu_pipeline_tree2_margin_guard, simulate_self_spec_gpu_pipeline_tree2_branch_guard, risk_offramp_margin, pipeline_mtp_k2_on_reject, simulate_self_spec_gpu_pipeline_reject_offramp_after)
                accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                draft_variant_note = simulate_self_spec_gpu_pipeline_draft_no_ffn ? " draft_no_ffn=1" : ""
                draft_no_ffn_layers_note = draft_no_ffn_layer_set ? " draft_no_ffn_layers=#{draft_no_ffn_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                draft_updown_note = pipeline_updown_rank ? " draft_pca_updown=#{pipeline_updown_rank}" : ""
                draft_updown_layers_note = (pipeline_updown_rank && draft_updown_layer_set) ? " draft_pca_updown_layers=#{draft_updown_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                draft_updown_fallback_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                draft_updown_warmup_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                draft_updown_margin_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_min_margin) ? " draft_pca_updown_min_margin=#{simulate_self_spec_gpu_pipeline_draft_updown_min_margin}" : ""
                split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                risk_offramp_note = risk_offramp_margin ? " risk_offramp_margin=#{risk_offramp_margin}" : ""
                tree2_note = (simulate_self_spec_gpu_pipeline_tree2_first || simulate_self_spec_gpu_pipeline_tree2_anywhere || simulate_self_spec_gpu_pipeline_tree2_staged_tokens > 0 || !simulate_self_spec_gpu_pipeline_tree2_margin_guard.nil? || !simulate_self_spec_gpu_pipeline_tree2_branch_guard.nil? || !risk_offramp_margin.nil? || !simulate_self_spec_gpu_pipeline_draft_updown_min_margin.nil? || simulate_self_spec_gpu_pipeline_mtp_k2_on_reject || simulate_self_spec_gpu_pipeline_reject_offramp_after > 0) ? self_spec_pipeline_tree2_note(pipe) : ""
                attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                agreement_note = simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate ? self_spec_pipeline_updown_agreement_note(pipe) : ""
                puts "self_spec_gpu_pipeline layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{pipeline_schedule.join(',')}#{split_note}#{draft_variant_note}#{draft_no_ffn_layers_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_layers_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{draft_updown_margin_note}#{risk_offramp_note}#{exact_refresh_note}#{backup_note}#{state_backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} draft_noffn_chunks=#{pipe[:draft_noffn_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{tree2_note}#{agreement_note}#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
                if dump_path = simulate_self_spec_gpu_pipeline_dump_cycles_path
                  schedule_label = "schedule=#{pipeline_schedule.join(',')}"
                  dump_self_spec_gpu_pipeline_cycles(dump_path.not_nil!, main_prompt_name, prompt, "self_lowrank/#{schedule_label}", simulate_logit_layers, rank, schedule_label, pipe)
                end
                append_draft_body_score(draft_body_score_rows, main_prompt_name, "schedule=#{pipeline_schedule.join(',')}", draft_split, pipeline_updown_rank, pipe, accept_rate)
                append_risk_offramp_score(risk_offramp_score_rows, main_prompt_name, "schedule=#{pipeline_schedule.join(',')}", draft_split, risk_offramp_margin, pipe, accept_rate)
              end
            end
          end
        end
      end
      unless simulate_self_spec_gpu_pipeline_suite_prompts.empty?
        simulate_self_spec_gpu_pipeline_suite_prompts.each do |suite_prompt|
          ProbeRuntime.self_spec_router_trace_label = suite_prompt[:name]
          suite_token_ids = token_ids_for_prompt(tok, suite_prompt[:text], tokens_limit)
          suite_calib_count = Math.min(calib_tokens, suite_token_ids.size - 1)
          raise "suite prompt #{suite_prompt[:name]} needs at least one held-out token" unless suite_calib_count > 0 && suite_calib_count < suite_token_ids.size
          suite_layer_vectors = {} of Int32 => BasisSet
          suite_layer_bases = {} of Int32 => BasisSet
          sorted_simulate_logit_layers.each do |il|
            vectors = recurrent_k_vectors_for_prompt(weights, suite_token_ids, il)
            suite_layer_vectors[il] = vectors
            suite_layer_bases[il] = vectors.map do |head_vectors|
              build_basis(head_vectors[0, suite_calib_count], max_rank, basis_mode, pca_iters)
            end
          end
          suite_rank_notes = sorted_simulate_logit_layers.map do |il|
            "#{il}:#{basis_rank_note(suite_layer_bases[il], rank)}"
          end
          puts "self_spec_gpu_pipeline_suite name=#{suite_prompt[:name]} token_vectors=#{suite_token_ids.size} calib_tokens=#{suite_calib_count} heldout_tokens=#{suite_token_ids.size - suite_calib_count} layer_basis_effective_ranks=#{suite_rank_notes.join(' ')}"
          suite_route_residual_stats = route_residual_stats(suite_layer_vectors, suite_layer_bases, rank, suite_calib_count, thresholds)
          suite_value_stats = self_spec_prompt_value_stats(suite_token_ids, suite_calib_count)
          if simulate_self_spec_gpu_pipeline_route_features
            puts prompt_route_feature_note(suite_prompt[:name], sorted_simulate_logit_layers, rank, suite_token_ids.size, suite_calib_count, suite_layer_vectors, suite_layer_bases, thresholds)
            prompt_route_layer_feature_notes(suite_prompt[:name], sorted_simulate_logit_layers, rank, suite_token_ids.size, suite_calib_count, suite_layer_vectors, suite_layer_bases, thresholds).each { |line| puts line }
          end
          if simulate_self_spec_gpu_pipeline_ffn_updown_route_features
            if adapters = ffn_updown_adapters
              if first_adapter = adapters.values.first?
                feature_rank = first_adapter.coeff_weights.size
                puts ffn_updown_route_feature_note(suite_prompt[:name], weights, suite_token_ids, suite_calib_count, sorted_simulate_logit_layers, adapters, feature_rank)
              end
            end
          end
          suite_residual_router_note = ""
          suite_residual_router_skip = false
          if residual_router_enabled
            suite_residual_router_stats = route_residual_stats(suite_layer_vectors, suite_layer_bases, rank, suite_calib_count, residual_router_thresholds)
            suite_residual_router_decision = self_spec_residual_router_decision(suite_residual_router_stats,
              simulate_self_spec_gpu_pipeline_residual_router_mean_max,
              simulate_self_spec_gpu_pipeline_residual_router_pass_threshold,
              simulate_self_spec_gpu_pipeline_residual_router_pass_rate_min)
            suite_residual_router_note = self_spec_residual_router_note(suite_residual_router_decision, suite_residual_router_stats,
              simulate_self_spec_gpu_pipeline_residual_router_mean_max,
              simulate_self_spec_gpu_pipeline_residual_router_pass_threshold,
              simulate_self_spec_gpu_pipeline_residual_router_pass_rate_min)
            suite_residual_router_skip = !suite_residual_router_decision[:run]
          end
          suite_value_router_note = ""
          suite_value_router_skip = false
          if value_router_enabled
            suite_value_router_stats = self_spec_prompt_value_stats(suite_token_ids, suite_calib_count)
            suite_value_router_decision = self_spec_value_router_decision(suite_value_router_stats,
              simulate_self_spec_gpu_pipeline_value_repeat_rate_min,
              simulate_self_spec_gpu_pipeline_value_bigram_repeat_rate_min,
              simulate_self_spec_gpu_pipeline_value_unique_rate_max)
            suite_value_router_note = self_spec_value_router_note(suite_value_router_decision, suite_value_router_stats,
              simulate_self_spec_gpu_pipeline_value_repeat_rate_min,
              simulate_self_spec_gpu_pipeline_value_bigram_repeat_rate_min,
              simulate_self_spec_gpu_pipeline_value_unique_rate_max)
            suite_value_router_skip = !suite_value_router_decision[:run]
          end
          suite_pre_submit_router_note = "#{suite_residual_router_note}#{suite_value_router_note}"
          suite_pre_submit_router_skip = suite_residual_router_skip || suite_value_router_skip
          run_noffn_fallback_abba.call("suite", suite_prompt[:name], suite_token_ids, suite_layer_bases)
          suite_route_selector_value = prompt_route_selector_feature_value(suite_route_residual_stats, suite_value_stats, route_selector_feature_name.to_s)
          if simulate_self_spec_gpu_pipeline_route_selector_abba > 0
            run_route_selector_abba.call("suite", suite_prompt[:name], suite_prompt[:text], suite_token_ids, suite_layer_bases, suite_route_selector_value)
          else
            run_route_selector.call("suite", suite_prompt[:name], suite_prompt[:text], suite_token_ids, suite_layer_bases, suite_route_selector_value, false, -1)
          end
          if simulate_self_spec_gpu_pipeline_suite_hybrid_sweep
            pipeline_gammas.each do |pipeline_gamma|
              pipeline_splits.each do |draft_split|
                pipeline_updown_options.each do |pipeline_updown_rank|
                  hybrid_routes.each do |route|
                    next if pipeline_updown_rank && route[:updown].nil?
                    next if pipeline_updown_rank.nil? && route[:updown]
                    route_updown_rank = route[:updown] ? pipeline_updown_rank : nil
                    pipe = simulate_self_spec_gpu_pipeline_run(weights, suite_token_ids, simulate_generate_tokens, pipeline_gamma, suite_layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, false, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, nil, route_updown_rank, ffn_updown_adapters, ffn_updown_adapter_q8_metal, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, simulate_self_spec_gpu_pipeline_draft_updown_min_margin, simulate_self_spec_gpu_pipeline_draft_updown_max_chunks, simulate_self_spec_gpu_pipeline_draft_updown_after_rejects, simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept, simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate, simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps, simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds, !simulate_self_spec_gpu_pipeline_legacy_full_state_backup, route[:noffn], route[:updown], simulate_self_spec_gpu_pipeline_tree2_first, simulate_self_spec_gpu_pipeline_tree2_anywhere, simulate_self_spec_gpu_pipeline_tree2_staged_tokens, simulate_self_spec_gpu_pipeline_tree2_margin_guard, simulate_self_spec_gpu_pipeline_tree2_branch_guard, simulate_self_spec_gpu_pipeline_risk_offramp_margin, pipeline_mtp_k2_on_reject, simulate_self_spec_gpu_pipeline_reject_offramp_after)
                    accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                    backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                    draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                    draft_updown_note = route_updown_rank ? " draft_pca_updown=#{route_updown_rank}" : ""
                    draft_updown_fallback_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                    draft_updown_warmup_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                    draft_updown_margin_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_min_margin) ? " draft_pca_updown_min_margin=#{simulate_self_spec_gpu_pipeline_draft_updown_min_margin}" : ""
                    split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                    route_note = hybrid_route_note(route, route_updown_rank)
                    tree2_note = (simulate_self_spec_gpu_pipeline_tree2_first || simulate_self_spec_gpu_pipeline_tree2_anywhere || simulate_self_spec_gpu_pipeline_tree2_staged_tokens > 0 || !simulate_self_spec_gpu_pipeline_tree2_margin_guard.nil? || !simulate_self_spec_gpu_pipeline_tree2_branch_guard.nil? || !simulate_self_spec_gpu_pipeline_risk_offramp_margin.nil? || !simulate_self_spec_gpu_pipeline_draft_updown_min_margin.nil? || simulate_self_spec_gpu_pipeline_mtp_k2_on_reject || simulate_self_spec_gpu_pipeline_reject_offramp_after > 0) ? self_spec_pipeline_tree2_note(pipe) : ""
                    attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                    agreement_note = simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate ? self_spec_pipeline_updown_agreement_note(pipe) : ""
                    puts "self_spec_gpu_pipeline_suite_hybrid name=#{suite_prompt[:name]} layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{pipeline_gamma}#{split_note}#{route_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{draft_updown_margin_note}#{exact_refresh_note}#{backup_note}#{state_backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} draft_noffn_chunks=#{pipe[:draft_noffn_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{tree2_note}#{agreement_note}#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
                    if dump_path = simulate_self_spec_gpu_pipeline_dump_cycles_path
                      route_label = route_updown_rank ? "#{route[:name]}_updown#{route_updown_rank}" : route[:name]
                      dump_self_spec_gpu_pipeline_cycles(dump_path.not_nil!, suite_prompt[:name], suite_prompt[:text], "self_lowrank/gamma=#{pipeline_gamma}/route=#{route_label}", simulate_logit_layers, rank, "gamma=#{pipeline_gamma}", pipe)
                    end
                    append_route_score(route_score_rows, suite_prompt[:name], "gamma=#{pipeline_gamma}", route, draft_split, route_updown_rank, pipe, accept_rate,
                      suite_route_residual_stats[:mean], suite_route_residual_stats[:p90], suite_route_residual_stats[:max],
                      suite_value_stats[:repeat_rate], suite_value_stats[:bigram_repeat_rate], suite_value_stats[:unique_rate])
                  end
                end
              end
            end
            simulate_self_spec_gpu_pipeline_schedules.each do |pipeline_schedule|
              next if pipeline_schedule.empty?
              pipeline_splits.each do |draft_split|
                pipeline_updown_options.each do |pipeline_updown_rank|
                  hybrid_routes.each do |route|
                    next if pipeline_updown_rank && route[:updown].nil?
                    next if pipeline_updown_rank.nil? && route[:updown]
                    route_updown_rank = route[:updown] ? pipeline_updown_rank : nil
                    pipe = simulate_self_spec_gpu_pipeline_run(weights, suite_token_ids, simulate_generate_tokens, pipeline_schedule[0], suite_layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, false, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, pipeline_schedule, route_updown_rank, ffn_updown_adapters, ffn_updown_adapter_q8_metal, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, simulate_self_spec_gpu_pipeline_draft_updown_min_margin, simulate_self_spec_gpu_pipeline_draft_updown_max_chunks, simulate_self_spec_gpu_pipeline_draft_updown_after_rejects, simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept, simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate, simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps, simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds, !simulate_self_spec_gpu_pipeline_legacy_full_state_backup, route[:noffn], route[:updown], simulate_self_spec_gpu_pipeline_tree2_first, simulate_self_spec_gpu_pipeline_tree2_anywhere, simulate_self_spec_gpu_pipeline_tree2_staged_tokens, simulate_self_spec_gpu_pipeline_tree2_margin_guard, simulate_self_spec_gpu_pipeline_tree2_branch_guard, simulate_self_spec_gpu_pipeline_risk_offramp_margin, pipeline_mtp_k2_on_reject, simulate_self_spec_gpu_pipeline_reject_offramp_after)
                    accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                    backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                    draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                    draft_updown_note = route_updown_rank ? " draft_pca_updown=#{route_updown_rank}" : ""
                    draft_updown_fallback_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                    draft_updown_warmup_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                    draft_updown_margin_note = (route_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_min_margin) ? " draft_pca_updown_min_margin=#{simulate_self_spec_gpu_pipeline_draft_updown_min_margin}" : ""
                    split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                    route_note = hybrid_route_note(route, route_updown_rank)
                    tree2_note = (simulate_self_spec_gpu_pipeline_tree2_first || simulate_self_spec_gpu_pipeline_tree2_anywhere || simulate_self_spec_gpu_pipeline_tree2_staged_tokens > 0 || !simulate_self_spec_gpu_pipeline_tree2_margin_guard.nil? || !simulate_self_spec_gpu_pipeline_tree2_branch_guard.nil? || !simulate_self_spec_gpu_pipeline_risk_offramp_margin.nil? || !simulate_self_spec_gpu_pipeline_draft_updown_min_margin.nil? || simulate_self_spec_gpu_pipeline_mtp_k2_on_reject || simulate_self_spec_gpu_pipeline_reject_offramp_after > 0) ? self_spec_pipeline_tree2_note(pipe) : ""
                    attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                    agreement_note = simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate ? self_spec_pipeline_updown_agreement_note(pipe) : ""
                    puts "self_spec_gpu_pipeline_suite_hybrid name=#{suite_prompt[:name]} layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{pipeline_schedule.join(',')}#{split_note}#{route_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{draft_updown_margin_note}#{exact_refresh_note}#{backup_note}#{state_backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} draft_noffn_chunks=#{pipe[:draft_noffn_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{tree2_note}#{agreement_note}#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
                    if dump_path = simulate_self_spec_gpu_pipeline_dump_cycles_path
                      route_label = route_updown_rank ? "#{route[:name]}_updown#{route_updown_rank}" : route[:name]
                      schedule_label = "schedule=#{pipeline_schedule.join(',')}"
                      dump_self_spec_gpu_pipeline_cycles(dump_path.not_nil!, suite_prompt[:name], suite_prompt[:text], "self_lowrank/#{schedule_label}/route=#{route_label}", simulate_logit_layers, rank, schedule_label, pipe)
                    end
                    append_route_score(route_score_rows, suite_prompt[:name], "schedule=#{pipeline_schedule.join(',')}", route, draft_split, route_updown_rank, pipe, accept_rate,
                      suite_route_residual_stats[:mean], suite_route_residual_stats[:p90], suite_route_residual_stats[:max],
                      suite_value_stats[:repeat_rate], suite_value_stats[:bigram_repeat_rate], suite_value_stats[:unique_rate])
                  end
                end
              end
            end
            next
          end
          pipeline_gammas.each do |pipeline_gamma|
            pipeline_splits.each do |draft_split|
              pipeline_updown_options.each do |pipeline_updown_rank|
                risk_offramp_options.each do |risk_offramp_margin|
                  branch_snapshot_mode_options.each do |branch_snapshot_mode|
                  apply_branch_snapshot_mode.call(branch_snapshot_mode)
                  if suite_pre_submit_router_skip
                    backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                    draft_variant_note = simulate_self_spec_gpu_pipeline_draft_no_ffn ? " draft_no_ffn=1" : ""
                    draft_no_ffn_layers_note = draft_no_ffn_layer_set ? " draft_no_ffn_layers=#{draft_no_ffn_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                    draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                    draft_updown_note = pipeline_updown_rank ? " draft_pca_updown=#{pipeline_updown_rank}" : ""
                    draft_updown_layers_note = (pipeline_updown_rank && draft_updown_layer_set) ? " draft_pca_updown_layers=#{draft_updown_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                    draft_updown_fallback_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                    draft_updown_warmup_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                    draft_updown_margin_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_min_margin) ? " draft_pca_updown_min_margin=#{simulate_self_spec_gpu_pipeline_draft_updown_min_margin}" : ""
                    split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                    risk_offramp_note = risk_offramp_margin ? " risk_offramp_margin=#{risk_offramp_margin}" : ""
                    branch_snapshot_mode_note = branch_snapshot_mode[:name].empty? ? "" : " branch_snapshot_mode=#{branch_snapshot_mode[:name]}"
                    exact = simulate_self_spec_gpu_pipeline_exact_fallback_run(weights, suite_token_ids, simulate_generate_tokens)
                    puts "self_spec_gpu_pipeline_suite name=#{suite_prompt[:name]} layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{pipeline_gamma}#{branch_snapshot_mode_note}#{split_note}#{draft_variant_note}#{draft_no_ffn_layers_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_layers_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{draft_updown_margin_note}#{risk_offramp_note}#{suite_pre_submit_router_note}#{exact_refresh_note}#{backup_note}#{state_backup_note}#{self_spec_pipeline_exact_fallback_fields(exact, simulate_generate_tokens)}"
                    next
                  end
                  pipe = simulate_self_spec_gpu_pipeline_run(weights, suite_token_ids, simulate_generate_tokens, pipeline_gamma, suite_layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, simulate_self_spec_gpu_pipeline_draft_no_ffn, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, nil, pipeline_updown_rank, ffn_updown_adapters, ffn_updown_adapter_q8_metal, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, simulate_self_spec_gpu_pipeline_draft_updown_min_margin, simulate_self_spec_gpu_pipeline_draft_updown_max_chunks, simulate_self_spec_gpu_pipeline_draft_updown_after_rejects, simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept, simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate, simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps, simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds, !simulate_self_spec_gpu_pipeline_legacy_full_state_backup, draft_no_ffn_layer_set, draft_updown_layer_set, simulate_self_spec_gpu_pipeline_tree2_first, simulate_self_spec_gpu_pipeline_tree2_anywhere, simulate_self_spec_gpu_pipeline_tree2_staged_tokens, simulate_self_spec_gpu_pipeline_tree2_margin_guard, simulate_self_spec_gpu_pipeline_tree2_branch_guard, risk_offramp_margin, pipeline_mtp_k2_on_reject, simulate_self_spec_gpu_pipeline_reject_offramp_after)
                  accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                  backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                  draft_variant_note = simulate_self_spec_gpu_pipeline_draft_no_ffn ? " draft_no_ffn=1" : ""
                  draft_no_ffn_layers_note = draft_no_ffn_layer_set ? " draft_no_ffn_layers=#{draft_no_ffn_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                  draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                  draft_updown_note = pipeline_updown_rank ? " draft_pca_updown=#{pipeline_updown_rank}" : ""
                  draft_updown_layers_note = (pipeline_updown_rank && draft_updown_layer_set) ? " draft_pca_updown_layers=#{draft_updown_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                  draft_updown_fallback_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                  draft_updown_warmup_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                  draft_updown_margin_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_min_margin) ? " draft_pca_updown_min_margin=#{simulate_self_spec_gpu_pipeline_draft_updown_min_margin}" : ""
                  split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                  risk_offramp_note = risk_offramp_margin ? " risk_offramp_margin=#{risk_offramp_margin}" : ""
                  branch_snapshot_mode_note = branch_snapshot_mode[:name].empty? ? "" : " branch_snapshot_mode=#{branch_snapshot_mode[:name]}"
                  branch_snapshot_mode_score_note = branch_snapshot_mode[:name].empty? ? "" : "/branch_snapshot=#{branch_snapshot_mode[:name]}"
                  tree2_note = (simulate_self_spec_gpu_pipeline_tree2_first || simulate_self_spec_gpu_pipeline_tree2_anywhere || simulate_self_spec_gpu_pipeline_tree2_staged_tokens > 0 || !simulate_self_spec_gpu_pipeline_tree2_margin_guard.nil? || !simulate_self_spec_gpu_pipeline_tree2_branch_guard.nil? || !risk_offramp_margin.nil? || !simulate_self_spec_gpu_pipeline_draft_updown_min_margin.nil? || simulate_self_spec_gpu_pipeline_mtp_k2_on_reject || simulate_self_spec_gpu_pipeline_reject_offramp_after > 0) ? self_spec_pipeline_tree2_note(pipe) : ""
                  attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                  agreement_note = simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate ? self_spec_pipeline_updown_agreement_note(pipe) : ""
                  puts "self_spec_gpu_pipeline_suite name=#{suite_prompt[:name]} layers=#{simulate_logit_layers.join(',')} rank=#{rank} gamma=#{pipeline_gamma}#{branch_snapshot_mode_note}#{split_note}#{draft_variant_note}#{draft_no_ffn_layers_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_layers_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{draft_updown_margin_note}#{risk_offramp_note}#{suite_pre_submit_router_note}#{exact_refresh_note}#{backup_note}#{state_backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} draft_noffn_chunks=#{pipe[:draft_noffn_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{tree2_note}#{agreement_note}#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
                  if dump_path = simulate_self_spec_gpu_pipeline_dump_cycles_path
                    dump_self_spec_gpu_pipeline_cycles(dump_path.not_nil!, suite_prompt[:name], suite_prompt[:text], "self_lowrank/gamma=#{pipeline_gamma}", simulate_logit_layers, rank, "gamma=#{pipeline_gamma}", pipe)
                  end
                  append_draft_body_score(draft_body_score_rows, suite_prompt[:name], "gamma=#{pipeline_gamma}#{branch_snapshot_mode_score_note}", draft_split, pipeline_updown_rank, pipe, accept_rate)
                  append_risk_offramp_score(risk_offramp_score_rows, suite_prompt[:name], "gamma=#{pipeline_gamma}#{branch_snapshot_mode_score_note}", draft_split, risk_offramp_margin, pipe, accept_rate)
                end
              end
            end
          end
          end
          simulate_self_spec_gpu_pipeline_schedules.each do |pipeline_schedule|
            next if pipeline_schedule.empty?
            pipeline_splits.each do |draft_split|
              pipeline_updown_options.each do |pipeline_updown_rank|
                risk_offramp_options.each do |risk_offramp_margin|
                  pipe = simulate_self_spec_gpu_pipeline_run(weights, suite_token_ids, simulate_generate_tokens, pipeline_schedule[0], suite_layer_bases, rank, !simulate_self_spec_gpu_pipeline_no_backup, draft_split, simulate_self_spec_gpu_pipeline_draft_no_ffn, simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn, pipeline_schedule, pipeline_updown_rank, ffn_updown_adapters, ffn_updown_adapter_q8_metal, simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject, simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts, simulate_self_spec_gpu_pipeline_draft_updown_min_margin, simulate_self_spec_gpu_pipeline_draft_updown_max_chunks, simulate_self_spec_gpu_pipeline_draft_updown_after_rejects, simulate_self_spec_gpu_pipeline_draft_updown_refresh_on_accept, simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate, simulate_self_spec_gpu_pipeline_draft_updown_agreement_steps, simulate_self_spec_gpu_pipeline_draft_updown_agreement_margin_thresholds, !simulate_self_spec_gpu_pipeline_legacy_full_state_backup, draft_no_ffn_layer_set, draft_updown_layer_set, simulate_self_spec_gpu_pipeline_tree2_first, simulate_self_spec_gpu_pipeline_tree2_anywhere, simulate_self_spec_gpu_pipeline_tree2_staged_tokens, simulate_self_spec_gpu_pipeline_tree2_margin_guard, simulate_self_spec_gpu_pipeline_tree2_branch_guard, risk_offramp_margin, pipeline_mtp_k2_on_reject, simulate_self_spec_gpu_pipeline_reject_offramp_after)
                  accept_rate = pipe[:proposed_tokens] > 0 ? (100.0 * pipe[:accepted_draft_tokens] / pipe[:proposed_tokens]) : 0.0
                  backup_note = simulate_self_spec_gpu_pipeline_no_backup ? " no_backup=1" : ""
                  draft_variant_note = simulate_self_spec_gpu_pipeline_draft_no_ffn ? " draft_no_ffn=1" : ""
                  draft_no_ffn_layers_note = draft_no_ffn_layer_set ? " draft_no_ffn_layers=#{draft_no_ffn_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                  draft_skip_rec_note = simulate_self_spec_gpu_pipeline_draft_skip_recurrent_ffn ? " draft_skip_recurrent_ffn=1" : ""
                  draft_updown_note = pipeline_updown_rank ? " draft_pca_updown=#{pipeline_updown_rank}" : ""
                  draft_updown_layers_note = (pipeline_updown_rank && draft_updown_layer_set) ? " draft_pca_updown_layers=#{draft_updown_layer_set.not_nil!.to_a.sort.join(',')}" : ""
                  draft_updown_fallback_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_fallback_on_reject) ? " draft_pca_updown_fallback=reject" : ""
                  draft_updown_warmup_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts > 0) ? " draft_pca_updown_after_full_accepts=#{simulate_self_spec_gpu_pipeline_draft_updown_after_full_accepts}" : ""
                  draft_updown_margin_note = (pipeline_updown_rank && simulate_self_spec_gpu_pipeline_draft_updown_min_margin) ? " draft_pca_updown_min_margin=#{simulate_self_spec_gpu_pipeline_draft_updown_min_margin}" : ""
                  split_note = draft_split.nil? ? "" : " draft_split=#{draft_split}"
                  risk_offramp_note = risk_offramp_margin ? " risk_offramp_margin=#{risk_offramp_margin}" : ""
                  tree2_note = (simulate_self_spec_gpu_pipeline_tree2_first || simulate_self_spec_gpu_pipeline_tree2_anywhere || simulate_self_spec_gpu_pipeline_tree2_staged_tokens > 0 || !simulate_self_spec_gpu_pipeline_tree2_margin_guard.nil? || !simulate_self_spec_gpu_pipeline_tree2_branch_guard.nil? || !risk_offramp_margin.nil? || !simulate_self_spec_gpu_pipeline_draft_updown_min_margin.nil? || simulate_self_spec_gpu_pipeline_mtp_k2_on_reject || simulate_self_spec_gpu_pipeline_reject_offramp_after > 0) ? self_spec_pipeline_tree2_note(pipe) : ""
                  attr_note = simulate_self_spec_gpu_pipeline_attribution ? self_spec_pipeline_attr_note(pipe) : ""
                  agreement_note = simulate_self_spec_gpu_pipeline_draft_updown_agreement_gate ? self_spec_pipeline_updown_agreement_note(pipe) : ""
                  puts "self_spec_gpu_pipeline_suite name=#{suite_prompt[:name]} layers=#{simulate_logit_layers.join(',')} rank=#{rank} schedule=#{pipeline_schedule.join(',')}#{split_note}#{draft_variant_note}#{draft_no_ffn_layers_note}#{draft_skip_rec_note}#{draft_updown_note}#{draft_updown_layers_note}#{draft_updown_fallback_note}#{draft_updown_warmup_note}#{draft_updown_margin_note}#{risk_offramp_note}#{exact_refresh_note}#{backup_note}#{state_backup_note} gen_tokens=#{simulate_generate_tokens} chunks=#{pipe[:chunks]} draft_updown_chunks=#{pipe[:draft_updown_chunks]} draft_noffn_chunks=#{pipe[:draft_noffn_chunks]} rejections=#{pipe[:rejections]} accepted_draft_tokens=#{pipe[:accepted_draft_tokens]} proposed_tokens=#{pipe[:proposed_tokens]} accept_rate=#{accept_rate.round(2)}% parity=#{pipe[:parity]} gamma_history=#{pipe[:gamma_history].join(',')} draft_seed_ms=#{pipe[:draft_seed_ms].round(3)} draft_next_ms=#{pipe[:draft_next_ms].round(3)} verifier_ms=#{pipe[:verifier_ms].round(3)} draft_wait_ms=#{pipe[:draft_wait_ms].round(3)} backup_ms=#{pipe[:backup_ms].round(3)} rebuild_ms=#{pipe[:rebuild_ms].round(3)} controller_ms=#{pipe[:controller_ms].round(3)} replay_ms=#{pipe[:replay_ms].round(3)} plain_exact_ms=#{pipe[:plain_exact_ms].round(3)} serial_ms=#{pipe[:serial_ms].round(3)} overlap_ms=#{pipe[:overlap_ms].round(3)} hidden_ms=#{pipe[:hidden_ms].round(3)} speedup=#{pipe[:speedup].round(4)}x plain_speedup=#{pipe[:plain_speedup].round(4)}x#{tree2_note}#{agreement_note}#{attr_note} exact_ids=#{pipe[:exact_ids].join(',')} emitted_ids=#{pipe[:emitted_ids].join(',')}"
                  if dump_path = simulate_self_spec_gpu_pipeline_dump_cycles_path
                    schedule_label = "schedule=#{pipeline_schedule.join(',')}"
                    dump_self_spec_gpu_pipeline_cycles(dump_path.not_nil!, suite_prompt[:name], suite_prompt[:text], "self_lowrank/#{schedule_label}", simulate_logit_layers, rank, schedule_label, pipe)
                  end
                  append_draft_body_score(draft_body_score_rows, suite_prompt[:name], "schedule=#{pipeline_schedule.join(',')}", draft_split, pipeline_updown_rank, pipe, accept_rate)
                  append_risk_offramp_score(risk_offramp_score_rows, suite_prompt[:name], "schedule=#{pipeline_schedule.join(',')}", draft_split, risk_offramp_margin, pipe, accept_rate)
                end
              end
            end
          end
        end
      end
      if simulate_self_spec_gpu_pipeline_hybrid_sweep && (simulate_self_spec_gpu_pipeline_route_scoreboard || simulate_self_spec_gpu_pipeline_hybrid_rich_sweep || simulate_self_spec_gpu_pipeline_suite_hybrid_sweep)
        print_route_scoreboard(route_score_rows)
        if simulate_self_spec_gpu_pipeline_suite_hybrid_sweep
          print_route_stability_scoreboard(route_score_rows)
          print_route_oracle_scoreboard(route_score_rows)
          print_route_selector_scoreboard(route_score_rows)
        end
      end
      if !draft_body_score_rows.empty? && (simulate_self_spec_gpu_pipeline_draft_updown_rank || !simulate_self_spec_gpu_pipeline_draft_updown_ranks.empty?)
        print_draft_body_scoreboard(draft_body_score_rows)
        print_draft_body_stability_scoreboard(draft_body_score_rows)
      end
      if !risk_offramp_score_rows.empty? && (!simulate_self_spec_gpu_pipeline_risk_offramp_margins.empty? || simulate_self_spec_gpu_pipeline_risk_offramp_margin)
        print_risk_offramp_scoreboard(risk_offramp_score_rows)
        print_risk_offramp_stability_scoreboard(risk_offramp_score_rows)
      end
    end
    if (simulate_self_spec_gpu_pipeline_route_features || simulate_self_spec_gpu_pipeline_ffn_updown_route_features) && !pipeline_route_active && !simulate_self_spec_gpu_pipeline_suite_prompts.empty?
      simulate_self_spec_gpu_pipeline_suite_prompts.each do |suite_prompt|
        suite_token_ids = token_ids_for_prompt(tok, suite_prompt[:text], tokens_limit)
        suite_calib_count = Math.min(calib_tokens, suite_token_ids.size - 1)
        raise "suite prompt #{suite_prompt[:name]} needs at least one held-out token" unless suite_calib_count > 0 && suite_calib_count < suite_token_ids.size
        suite_layer_vectors = {} of Int32 => BasisSet
        suite_layer_bases = {} of Int32 => BasisSet
        sorted_simulate_logit_layers.each do |il|
          vectors = recurrent_k_vectors_for_prompt(weights, suite_token_ids, il)
          suite_layer_vectors[il] = vectors
          suite_layer_bases[il] = vectors.map do |head_vectors|
            build_basis(head_vectors[0, suite_calib_count], max_rank, basis_mode, pca_iters)
          end
        end
        suite_rank_notes = sorted_simulate_logit_layers.map do |il|
          "#{il}:#{basis_rank_note(suite_layer_bases[il], rank)}"
        end
        puts "self_spec_gpu_pipeline_suite name=#{suite_prompt[:name]} token_vectors=#{suite_token_ids.size} calib_tokens=#{suite_calib_count} heldout_tokens=#{suite_token_ids.size - suite_calib_count} layer_basis_effective_ranks=#{suite_rank_notes.join(' ')}"
        if simulate_self_spec_gpu_pipeline_route_features
          puts prompt_route_feature_note(suite_prompt[:name], sorted_simulate_logit_layers, rank, suite_token_ids.size, suite_calib_count, suite_layer_vectors, suite_layer_bases, thresholds)
          prompt_route_layer_feature_notes(suite_prompt[:name], sorted_simulate_logit_layers, rank, suite_token_ids.size, suite_calib_count, suite_layer_vectors, suite_layer_bases, thresholds).each { |line| puts line }
        end
        if simulate_self_spec_gpu_pipeline_ffn_updown_route_features
          if adapters = ffn_updown_adapters
            if first_adapter = adapters.values.first?
              feature_rank = first_adapter.coeff_weights.size
              puts ffn_updown_route_feature_note(suite_prompt[:name], weights, suite_token_ids, suite_calib_count, sorted_simulate_logit_layers, adapters, feature_rank)
            end
          end
        end
      end
    end
  end
end

ranks.each do |rank|
  all_residuals = [] of Float64
  per_head.each_with_index do |vectors, head|
    basis = bases[head]
    vectors[calib_count, vectors.size - calib_count].each do |v|
      all_residuals << residual_norm(v, basis, rank)
    end
  end

  sorted = all_residuals.sort
  mean = all_residuals.sum / all_residuals.size
  p50 = sorted[sorted.size // 2]
  p90 = sorted[(sorted.size * 90 // 100).clamp(0, sorted.size - 1)]
  p99 = sorted[(sorted.size * 99 // 100).clamp(0, sorted.size - 1)]
  max = sorted[-1]
  pass = thresholds.map do |threshold|
    passed = all_residuals.count { |r| r <= threshold }
    "#{threshold.round(4)}:#{(100.0 * passed / all_residuals.size).round(2)}%"
  end
  line = "rank=#{rank} mean_residual=#{mean.round(6)} p50=#{p50.round(6)} p90=#{p90.round(6)} p99=#{p99.round(6)} max=#{max.round(6)} pass_rates=#{pass.join(',')}"
  if simulate_delta
    hp = weights.hparams
    drift = simulate_projected_delta(samples, bases, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " y_rmse=#{drift[:y_rmse].round(6)} y_max=#{drift[:y_max].round(6)} state_rmse=#{drift[:state_rmse].round(6)} state_max=#{drift[:state_max].round(6)}"
  end
  if simulate_lowrank
    hp = weights.hparams
    lr = simulate_lowrank_projected_delta(samples, bases, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " lr_exact_y_rmse=#{lr[:exact_y_rmse].round(6)} lr_exact_y_max=#{lr[:exact_y_max].round(6)} lr_proof_y_rmse=#{lr[:proof_y_rmse].round(8)} lr_proof_y_max=#{lr[:proof_y_max].round(8)} lr_proof_state_rmse=#{lr[:proof_state_rmse].round(8)} lr_proof_state_max=#{lr[:proof_state_max].round(8)}"
  end
  if simulate_lowrank_metal
    hp = weights.hparams
    lr_metal = simulate_lowrank_projected_delta_metal(samples, bases, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " lr_metal_steps=#{lr_metal[:steps]} lr_cpu_ms=#{lr_metal[:cpu_ms].round(3)} lr_metal_ms=#{lr_metal[:metal_ms].round(3)} lr_metal_y_rmse=#{lr_metal[:y_rmse].round(8)} lr_metal_y_max=#{lr_metal[:y_max].round(8)} lr_metal_state_rmse=#{lr_metal[:state_rmse].round(8)} lr_metal_state_max=#{lr_metal[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_project
    hp = weights.hparams
    lr_project = simulate_lowrank_projected_delta_metal_project(samples, bases, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " lr_project_steps=#{lr_project[:steps]} lr_project_cpu_ms=#{lr_project[:cpu_ms].round(3)} lr_project_metal_ms=#{lr_project[:metal_ms].round(3)} lr_project_y_rmse=#{lr_project[:y_rmse].round(8)} lr_project_y_max=#{lr_project[:y_max].round(8)} lr_project_state_rmse=#{lr_project[:state_rmse].round(8)} lr_project_state_max=#{lr_project[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_chunk
    hp = weights.hparams
    lr_chunk = simulate_lowrank_projected_delta_metal_chunk(samples, bases, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " lr_chunk_steps=#{lr_chunk[:steps]} lr_chunk_cpu_ms=#{lr_chunk[:cpu_ms].round(3)} lr_chunk_metal_ms=#{lr_chunk[:metal_ms].round(3)} lr_chunk_y_rmse=#{lr_chunk[:y_rmse].round(8)} lr_chunk_y_max=#{lr_chunk[:y_max].round(8)} lr_chunk_state_rmse=#{lr_chunk[:state_rmse].round(8)} lr_chunk_state_max=#{lr_chunk[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_chunk_out
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_chunk_out = simulate_lowrank_projected_delta_metal_chunk_out(samples, bases, target_layer.ssm_out_qw, target_layer.ssm_norm, hp.rms_eps.to_f32, rank, calib_count,
      hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size)
    line += " lr_chunk_out_steps=#{lr_chunk_out[:steps]} lr_chunk_out_cpu_ms=#{lr_chunk_out[:cpu_ms].round(3)} lr_chunk_out_metal_ms=#{lr_chunk_out[:metal_ms].round(3)} lr_chunk_out_rmse=#{lr_chunk_out[:out_rmse].round(8)} lr_chunk_out_max=#{lr_chunk_out[:out_max].round(8)} lr_chunk_out_state_rmse=#{lr_chunk_out[:state_rmse].round(8)} lr_chunk_out_state_max=#{lr_chunk_out[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_layer_chunk
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_layer = simulate_lowrank_recurrent_layer_metal_chunk(samples, bases, target_layer, hp, rank, calib_count)
    line += " lr_layer_chunk_steps=#{lr_layer[:steps]} lr_layer_chunk_cpu_ms=#{lr_layer[:cpu_ms].round(3)} lr_layer_chunk_metal_ms=#{lr_layer[:metal_ms].round(3)} lr_layer_chunk_rmse=#{lr_layer[:layer_rmse].round(8)} lr_layer_chunk_max=#{lr_layer[:layer_max].round(8)} lr_layer_chunk_state_rmse=#{lr_layer[:state_rmse].round(8)} lr_layer_chunk_state_max=#{lr_layer[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_layer_full
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_full = simulate_lowrank_recurrent_layer_full_metal_chunk(samples, bases, target_layer, hp, rank, calib_count)
    line += " lr_layer_full_steps=#{lr_full[:steps]} lr_layer_full_cpu_ms=#{lr_full[:cpu_ms].round(3)} lr_layer_full_metal_ms=#{lr_full[:metal_ms].round(3)} lr_layer_full_rmse=#{lr_full[:layer_rmse].round(8)} lr_layer_full_max=#{lr_full[:layer_max].round(8)} lr_layer_full_state_rmse=#{lr_full[:state_rmse].round(8)} lr_layer_full_state_max=#{lr_full[:state_max].round(8)}"
  end
  if layer_updown_rank = simulate_lowrank_metal_layer_updown_rank
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    ffn_vectors = ffn_activation_vectors_for_prompt(weights, token_ids, [layer_index], calib_count)[layer_index]
    ffn_basis = pca_basis(ffn_vectors, layer_updown_rank, pca_iters)
    down_basis = ffn_basis.map do |basis_vec|
      ML::GGUF::Qwen35CPU.qmatvec_nobias(target_layer.ffn_down_qw, basis_vec.map(&.to_f32))
    end
    updown_samples = ffn_updown_samples_for_token_sets(weights, [token_ids[0, calib_count]], [layer_index], calib_count)[layer_index]
    updown_adapter = train_ffn_updown_adapter(updown_samples, ffn_basis, down_basis, layer_updown_rank)
    lr_updown = simulate_lowrank_recurrent_layer_updown_metal_chunk(samples, bases, target_layer, hp, rank, calib_count, updown_adapter, layer_updown_rank)
    line += " lr_layer_updown_steps=#{lr_updown[:steps]} lr_layer_updown_rank=#{lr_updown[:updown_rank]} lr_layer_updown_cpu_ms=#{lr_updown[:cpu_ms].round(3)} lr_layer_updown_metal_ms=#{lr_updown[:metal_ms].round(3)} lr_layer_updown_rmse=#{lr_updown[:layer_rmse].round(8)} lr_layer_updown_max=#{lr_updown[:layer_max].round(8)} lr_layer_updown_state_rmse=#{lr_updown[:state_rmse].round(8)} lr_layer_updown_state_max=#{lr_updown[:state_max].round(8)}"
  end
  if simulate_lowrank_metal_layer_overlap
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_overlap = simulate_lowrank_recurrent_layer_full_async_overlap(samples, bases, target_layer, hp, rank, calib_count)
    line += " lr_layer_overlap_steps=#{lr_overlap[:steps]} lr_layer_overlap_serial_ms=#{lr_overlap[:serial_ms].round(3)} lr_layer_overlap_async_ms=#{lr_overlap[:async_ms].round(3)} lr_layer_overlap_speedup=#{lr_overlap[:speedup].round(4)} lr_layer_overlap_output_max=#{lr_overlap[:output_max].round(8)}"
  end
  if simulate_lowrank_metal_verifier_overlap
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_verify = simulate_lowrank_draft_exact_verifier_overlap(samples, bases, weights, token_ids, target_layer, hp, rank, calib_count)
    line += " lr_verifier_overlap_steps=#{lr_verify[:steps]} lr_verifier_draft_ms=#{lr_verify[:draft_ms].round(3)} lr_verifier_verify_ms=#{lr_verify[:verifier_ms].round(3)} lr_verifier_serial_ms=#{lr_verify[:serial_ms].round(3)} lr_verifier_overlap_ms=#{lr_verify[:overlap_ms].round(3)} lr_verifier_speedup=#{lr_verify[:speedup].round(4)} lr_verifier_hidden_ms=#{lr_verify[:hidden_ms].round(3)} lr_verifier_draft_output_max=#{lr_verify[:draft_output_max].round(8)} lr_verifier_match=#{lr_verify[:verifier_match]}"
  end
  if simulate_lowrank_metal_decode_verifier_overlap
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    lr_decode_verify = simulate_lowrank_draft_exact_decode_verifier_overlap(samples, bases, weights, token_ids, target_layer, hp, rank, calib_count)
    line += " lr_decode_verify_steps=#{lr_decode_verify[:steps]} lr_decode_verify_draft_ms=#{lr_decode_verify[:draft_ms].round(3)} lr_decode_verify_serial_ms=#{lr_decode_verify[:verifier_serial_ms].round(3)} lr_decode_verify_async_ms=#{lr_decode_verify[:verifier_async_ms].round(3)} lr_decode_verify_overlap_ms=#{lr_decode_verify[:overlap_ms].round(3)} lr_decode_verify_async_speedup=#{lr_decode_verify[:async_speedup].round(4)} lr_decode_verify_overlap_speedup=#{lr_decode_verify[:overlap_speedup].round(4)} lr_decode_verify_hidden_ms=#{lr_decode_verify[:hidden_ms].round(3)} lr_decode_verify_draft_output_max=#{lr_decode_verify[:draft_output_max].round(8)} lr_decode_verify_match=#{lr_decode_verify[:verifier_match]}"
  end
  if simulate_exact_verifier_ltp
    ltp = simulate_exact_verifier_ltp_proxy(weights, token_ids, calib_count)
    line += " exact_ltp_steps=#{ltp[:steps]} exact_ltp_decode_serial_ms=#{ltp[:decode_serial_ms].round(3)} exact_ltp_decode_queued_ms=#{ltp[:decode_queued_ms].round(3)} exact_ltp_chunk_major_ms=#{ltp[:chunk_major_ms].round(3)} exact_ltp_queued_speedup=#{ltp[:queued_speedup].round(4)} exact_ltp_speedup=#{ltp[:ltp_speedup].round(4)} exact_ltp_queued_match=#{ltp[:queued_match]} exact_ltp_chunk_match=#{ltp[:chunk_match]}"
  end
  if simulate_lowrank_metal_chunk_thread_overlap
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    ltp_overlap = simulate_lowrank_draft_exact_chunk_verifier_thread_overlap(samples, bases, weights, token_ids, target_layer, hp, rank, calib_count)
    line += " chunk_thread_steps=#{ltp_overlap[:steps]} chunk_thread_draft_ms=#{ltp_overlap[:draft_ms].round(3)} chunk_thread_verify_ms=#{ltp_overlap[:chunk_verifier_ms].round(3)} chunk_thread_serial_ms=#{ltp_overlap[:serial_ms].round(3)} chunk_thread_overlap_ms=#{ltp_overlap[:overlap_ms].round(3)} chunk_thread_speedup=#{ltp_overlap[:speedup].round(4)} chunk_thread_hidden_ms=#{ltp_overlap[:hidden_ms].round(3)} chunk_thread_draft_output_max=#{ltp_overlap[:draft_output_max].round(8)} chunk_thread_match=#{ltp_overlap[:verifier_match]}"
  end
  if simulate_multilayer_overlap_n > 0
    hp = weights.hparams
    target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer_index} is not recurrent"
    multi = simulate_lowrank_multilayer_chunk_thread_overlap(samples, bases, weights, token_ids, target_layer, hp, rank, calib_count, simulate_multilayer_overlap_n)
    line += " multi_thread_n_layers=#{multi[:n_layers]} multi_thread_steps=#{multi[:steps]} multi_thread_draft_ms=#{multi[:draft_ms].round(3)} multi_thread_draft_per_layer_ms=#{multi[:draft_per_layer_ms].round(3)} multi_thread_verify_ms=#{multi[:chunk_verifier_ms].round(3)} multi_thread_serial_ms=#{multi[:serial_ms].round(3)} multi_thread_overlap_ms=#{multi[:overlap_ms].round(3)} multi_thread_speedup=#{multi[:speedup].round(4)} multi_thread_hidden_ms=#{multi[:hidden_ms].round(3)} multi_thread_draft_output_max=#{multi[:draft_output_max].round(8)} multi_thread_match=#{multi[:verifier_match]}"
  end
  puts line
end
