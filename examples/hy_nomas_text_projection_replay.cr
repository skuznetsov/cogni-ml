require "option_parser"
require "json"
require "digest/sha256"
require "./support/hy_nomas_text_session"
require "./support/hy_nomas_probe_instrumentation"

module HyNomasTextProjectionReplay
  extend self

  REFERENCE_SHA256 = "167dd498c75efa396ca1f8ae33b68fa34368d2e4c96feb5d4de2d89c2666992e"
  MAX_NAMED_BYTES  = 1_500_000_000_i64
  ROWS             =                23
  INPUT_DIM        =              8192
  OUTPUT_DIM       =              2048
  INPUT_TENSOR     = "dit.txt.ff.activated"
  OUTPUT_TENSOR    = "dit.txt.ff.projected"

  def named_reserve_bytes : Int64
    (ROWS.to_i64 * INPUT_DIM * 4) +
      (ROWS.to_i64 * OUTPUT_DIM * 4) +
      (INPUT_DIM.to_i64 * OUTPUT_DIM * 4)
  end

  def validate_fixture_path!(path : String) : String
    raise ArgumentError.new("fixture must be a regular non-symlink file") unless File.file?(path) && !File.symlink?(path)
    resolved = File.realpath(path)
    raise ArgumentError.new("fixture exceeds the bounded manifest size") if File.size(resolved) > 2 * 1024 * 1024
    resolved
  end

  # Keep every borrowed resource reachable from before conversion dispatch
  # until explicit release after the caller's terminal drain. An undrained
  # owner remains rooted rather than letting a finalizer free staging early.
  class ProjectionRunOwner
    @@lifetime_mutex = Mutex.new
    @@live_owners = [] of ProjectionRunOwner

    @chain : ML::Metal::MPSCommandChain? = nil
    @chain_lease = HyNomasTextProbe::SessionChainLease(ML::Metal::MPSCommandChain).new
    @released = false

    def initialize(@checkpoint : ML::Video::CheckpointFile,
                   @weight_plan : ML::Video::WeightPlan,
                   @prepared : ML::Video::PreparedWeights,
                   @weight_arena : ML::Video::Arena,
                   @input_arena : ML::Video::Arena,
                   @output_arena : ML::Video::Arena,
                   @stage_arena : ML::Video::Arena,
                   @linear_plan : ML::Video::WanT5MPSLinearPlan)
    end

    def retain_lifetime_root! : Nil
      raise ArgumentError.new("projection run owner was released") if @released
      @@lifetime_mutex.synchronize do
        unless @@live_owners.any? { |current| current.same?(self) }
          @@live_owners << self
        end
      end
    end

    def attach_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("projection run owner already has a command chain") if @chain
      @chain = chain
    end

    def chain_attached? : Bool
      !@chain.nil?
    end

    def chain : ML::Metal::MPSCommandChain
      @chain || raise ArgumentError.new("projection run command chain is unavailable")
    end

    def begin_encode! : Nil
      @chain_lease.begin_encode!(chain)
    end

    def completed? : Bool
      if active_chain = @chain
        active_chain.completed?
      else
        false
      end
    end

    def release_before_submission! : Nil
      raise ArgumentError.new("cannot release an attached projection command chain before its terminal drain") if chain_attached?
      @chain_lease.ensure_terminal!
      release_owned_resources!
      @released = true
      release_lifetime_root!
    end

    def release_after_terminal! : Nil
      active_chain = chain
      @chain_lease.ensure_terminal!
      raise ArgumentError.new("projection command chain has not completed") unless active_chain.completed?
      release_owned_resources!
      active_chain.release
      @released = true
      release_lifetime_root!
    end

    private def release_owned_resources! : Nil
      @linear_plan.release
      @prepared.release
      {@stage_arena, @output_arena, @input_arena, @weight_arena}.each(&.release)
      @checkpoint.close
    end

    private def release_lifetime_root! : Nil
      @@lifetime_mutex.synchronize do
        @@live_owners.reject! { |current| current.same?(self) }
      end
    end
  end
end

reference = ""
reference_sha256 = ""
report = ""
budget = 0_i64
OptionParser.parse do |parser|
  parser.banner = "Replay the authenticated text FC2 projection only; no refiner or full block execution."
  parser.on("--reference PATH", "Authenticated text-conditioning fixture") { |value| reference = value }
  parser.on("--reference-sha256 SHA256", "Pinned text-conditioning fixture digest") { |value| reference_sha256 = value }
  parser.on("--report PATH", "Fresh post-terminal projection report") { |value| report = value }
  parser.on("--max-named-bytes N", "Aggregate named GPU-buffer budget (at most 1,500,000,000)") { |value| budget = value.to_i64 }
end

raise ArgumentError.new("reference, pinned digest, fresh report path and positive budget are required") if reference.empty? || reference_sha256.empty? || report.empty? || budget <= 0
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
raise ArgumentError.new("fixture digest is not the pinned authenticated reference") unless reference_sha256 == HyNomasTextProjectionReplay::REFERENCE_SHA256
raise ArgumentError.new("named GPU-buffer budget exceeds 1,500,000,000 bytes") if budget > HyNomasTextProjectionReplay::MAX_NAMED_BYTES
raise ArgumentError.new("refusing to replace projection report") if File.exists?(report) || File.symlink?(report)

reference = HyNomasTextProjectionReplay.validate_fixture_path!(reference)
raw_reference = File.read(reference)
observed_reference_sha256 = Digest::SHA256.hexdigest(raw_reference)
raise ArgumentError.new("text-conditioning fixture digest mismatch") unless observed_reference_sha256 == reference_sha256
document = JSON.parse(raw_reference)
qwen_reference_sha256 = document["qwen_source"]["reference_sha256"].as_s
fixture = HyNomasTextProbe::Fixture.new(reference, reference_sha256, qwen_reference_sha256)
raise ArgumentError.new("projection input shape mismatch") unless fixture.shape(HyNomasTextProjectionReplay::INPUT_TENSOR) == [1, 23, 8192]
raise ArgumentError.new("projection output shape mismatch") unless fixture.shape(HyNomasTextProjectionReplay::OUTPUT_TENSOR) == [1, 23, 2048]

config = HyNomasTextProbe::Session.config
rows = HyNomasTextProjectionReplay::ROWS
input_dim = HyNomasTextProjectionReplay::INPUT_DIM
output_dim = HyNomasTextProjectionReplay::OUTPUT_DIM
expected_sources = [
  "double_blocks.0.txt_mlp.fc2.weight",
  "double_blocks.0.txt_mlp.fc2.bias",
]
bindings = ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(
  config, ML::GGUF::SafeTensorDType::F32
).select { |binding| expected_sources.includes?(binding.source) }
raise ArgumentError.new("text FC2 checkpoint binding inventory mismatch") unless bindings.size == 2 && bindings.map(&.source).sort == expected_sources.sort

checkpoint : ML::Video::CheckpointFile? = nil
weight_plan : ML::Video::WeightPlan? = nil
weight_arena : ML::Video::Arena? = nil
input_arena : ML::Video::Arena? = nil
output_arena : ML::Video::Arena? = nil
stage_arena : ML::Video::Arena? = nil
prepared : ML::Video::PreparedWeights? = nil
linear_plan : ML::Video::WanT5MPSLinearPlan? = nil
owner : HyNomasTextProjectionReplay::ProjectionRunOwner? = nil

begin
  checkpoint = ML::Video::CheckpointFile.new(fixture.block_weights)
  reserve_bytes = HyNomasTextProjectionReplay.named_reserve_bytes
  weight_plan = ML::Video::WeightPlan.new(
    [checkpoint.not_nil!], bindings, budget,
    reserve_bytes: reserve_bytes,
    resident_dtype: ML::Video::WeightDType::BF16,
    require_all: false,
    allow_f32_to_bf16: true
  )
  named_bytes = weight_plan.not_nil!.peak_bytes
  raise ArgumentError.new("projection named-buffer plan exceeds the supplied budget") unless named_bytes <= budget

  weight_arena = ML::Video::Arena.new
  input_arena = ML::Video::Arena.new
  output_arena = ML::Video::Arena.new
  stage_arena = ML::Video::Arena.new
  prepared = ML::Video::PreparedWeights.new(weight_plan.not_nil!, weight_arena.not_nil!)
  input_values = fixture.values(HyNomasTextProjectionReplay::INPUT_TENSOR)
  input = input_arena.not_nil!.upload("txt_ff_activated", input_values)
  output = output_arena.not_nil!.alloc("txt_ff_projected", rows * output_dim)
  weight_stage = stage_arena.not_nil!.alloc("txt_fc2_weight_f32", input_dim * output_dim)
  weight = weight_arena.not_nil!["txt_mlp_fc2_weight"]
  bias = weight_arena.not_nil!["txt_mlp_fc2_bias"]

  ML::Video::MetalLinearMPS.prepare!
  plan_config = ML::Video::WanT5Config.new(
    vocab: 1, dim: 1, ffn_dim: 1, heads: 1, layers: 1,
    batches: 1, tokens: rows
  )
  linear_plan = ML::Video::WanT5MPSLinearPlan.new(
    plan_config, input_dim, output_dim, input, weight_stage, output
  )
  owner = HyNomasTextProjectionReplay::ProjectionRunOwner.new(
    checkpoint.not_nil!, weight_plan.not_nil!, prepared.not_nil!,
    weight_arena.not_nil!, input_arena.not_nil!, output_arena.not_nil!,
    stage_arena.not_nil!, linear_plan.not_nil!
  )
  active_owner = owner.not_nil!
  active_owner.retain_lifetime_root!
  active_owner.attach_chain!(ML::Metal::MPSCommandChain.new)
  active_chain = active_owner.chain
  before_allocations = ML::MetalBuffer.hy_text_probe_allocations
  before_readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  encode_started = Time.instant
  active_owner.begin_encode!
  prepared.not_nil!.encode(active_chain.encoder)
  ML::Video::MetalLinearMPS.project(
    active_chain.encoder, input, weight, bias, output,
    rows, input_dim, output_dim,
    weight_arena.not_nil!.dtype("txt_mlp_fc2_weight"),
    weight_arena.not_nil!.dtype("txt_mlp_fc2_bias"),
    linear_plan.not_nil!, weight_stage
  )
  encode_seconds = (Time.instant - encode_started).total_seconds
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - before_allocations
  prewait_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - before_readbacks
  raise "projection encoding allocated named GPU buffers" unless encode_allocations == 0
  raise "projection encoding read back activations" unless prewait_readbacks == 0
  raise "projection encoding performed an early device wait" unless active_chain.wait_count == 0

  wait_started = Time.instant
  active_chain.commit_and_wait
  wait_seconds = (Time.instant - wait_started).total_seconds
  raise "expected exactly one terminal wait" unless active_chain.wait_count == 1

  comparison = fixture.compare(output, HyNomasTextProjectionReplay::OUTPUT_TENSOR)
  diagnostic_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - before_readbacks
  raise "expected one post-wait output readback" unless diagnostic_readbacks == 1
  passed = comparison[:failures] == 0
  File.write(report, JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", "cogni_ml.hy_nomas_text_projection_replay"
      json.field "schema_version", 1
      json.field "reference", fixture.path
      json.field "reference_sha256", fixture.sha256
      json.field "block_weights", fixture.block_weights
      json.field "block_weights_sha256", fixture.document["weights"]["block"]["sha256"].as_s
      json.field "projection", "double_blocks.0.txt_mlp.fc2"
      json.field "linear_path", "ML::Video::MetalLinearMPS.project"
      json.field "input_scope", "authenticated saved dit.txt.ff.activated [1,23,8192]; preceding refiner and block work omitted"
      json.field "precision_scope", "F32 saved activation; BF16 resident weight and bias converted from authenticated F32 checkpoint tensors; F32 MPS output"
      json.field "named_bytes", named_bytes
      json.field "named_budget_bytes", budget
      json.field "opaque_mps_resources_excluded", true
      json.field "named_allocations_during_encode", encode_allocations
      json.field "activation_readbacks_before_terminal_wait", prewait_readbacks
      json.field "terminal_wait_count", active_chain.wait_count
      json.field "diagnostic_readbacks_after_terminal_wait", diagnostic_readbacks
      json.field "encode_seconds", encode_seconds
      json.field "terminal_wait_seconds", wait_seconds
      json.field "gpu_elapsed_seconds", active_chain.gpu_elapsed_seconds
      json.field "tolerance", {atol: HyNomasTextProbe::ATOL, rtol: HyNomasTextProbe::RTOL}
      json.field "comparison", comparison
      json.field "passed", passed
    end
  end)
  puts comparison.to_json
  raise "isolated text FC2 projection mismatch" unless passed
  puts "PASS: same-input text FC2 projection; one terminal wait"
ensure
  if active_owner = owner
    if active_owner.chain_attached?
      # Do not add a wait or release resources if terminal completion failed.
      # The owner stays in the setup-time root until process exit in that case.
      active_owner.release_after_terminal! if active_owner.completed?
    else
      active_owner.release_before_submission!
    end
  else
    # Constructors above run before chain submission, so partial setup is safe
    # to release directly when no complete owner was established.
    linear_plan.try(&.release)
    prepared.try(&.release)
    stage_arena.try(&.release)
    output_arena.try(&.release)
    input_arena.try(&.release)
    weight_arena.try(&.release)
    checkpoint.try(&.close)
  end
end
