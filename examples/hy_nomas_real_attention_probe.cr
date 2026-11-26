require "option_parser"
require "./support/hy_nomas_timeline_inputs"
require "./support/qwen25_prefix_probe"
require "../src/ml/video/hy_nomas_timeline_ops"

# Capture one actual first-step block, never a full DiT or media request. All
# neural operations share one chain; the sidecar is written only after drain.
manifest = ""
manifest_sha = ""
output = ""
budget = 0_i64
OptionParser.parse do |parser|
  parser.banner = "Bounded real-input Hy-Nomas first-block precision diagnostic."
  parser.on("--inputs PATH", "Authenticated continuation input manifest") { |v| manifest = v }
  parser.on("--inputs-sha256 SHA", "Trusted continuation manifest SHA-256") { |v| manifest_sha = v }
  parser.on("--output PATH", "New capture JSON and sibling tensor sidecar") { |v| output = v }
  parser.on("--max-named-bytes N", "Finite aggregate named GPU buffer budget") { |v| budget = v.to_i64 }
end
unless ARGV.empty? && !manifest.empty? && !manifest_sha.empty? &&
       !output.empty? && output.ends_with?(".json") && 0 < budget <= 8_i64 * 1024**3
  raise ArgumentError.new("inputs, digest, fresh .json output and a budget up to 8 GiB are required")
end
payload = output.sub(/\.json\z/, ".tensors.bin")
{output, payload}.each do |path|
  raise ArgumentError.new("refusing to replace capture #{path}") if File.exists?(path) || File.symlink?(path)
end

inputs = HyNomasGeneration::TimelineInputs.new(manifest, manifest_sha)
inputs.verify_payload!
base = inputs.base
config = HyNomasGeneration::CONFIG.compact_valid_text_prefix(base.valid_text_tokens, base.compact_text_tokens)
index = HyNomasGeneration.authenticate_bf16_index
pin = index.blocks.first
files = [] of ML::Video::CheckpointFile
arenas = [] of ML::Video::Arena
prepared = [] of ML::Video::PreparedWeights
patch : ML::Video::HyNomasPatchMPSPlan? = nil
refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
workspace : ML::Video::HyNomasMPSWorkspace? = nil
prefix : ML::Video::HyNomasTimelineOps::PrefixPlan? = nil
euler : ML::Video::HyNomasEulerPlan? = nil
chain : ML::Metal::MPSCommandChain? = nil
begin
  globals_file = ML::Video::BF16WeightCache.authenticate(index.globals_bf16_path, index.globals_bf16_sha256)
  files << globals_file
  block_file = ML::Video::BF16WeightCache.authenticate(pin.bf16_path, pin.bf16_sha256)
  files << block_file
  global_bindings = HyNomasQ4DiTIO::GLOBAL_BINDINGS.map do |b|
    ML::Video::WeightBinding.new(b.target, b.source, b.shape, ML::GGUF::SafeTensorDType::BF16)
  end
  globals_plan = ML::Video::WeightPlan.new([globals_file], global_bindings, budget,
    resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
  block_plan = ML::Video::WeightPlan.new([block_file],
    ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(config, ML::GGUF::SafeTensorDType::BF16),
    budget, resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
  raise "one-block diagnostic must not stage F32 weights" unless globals_plan.staging_bytes == 0 && block_plan.staging_bytes == 0
  # Include all known named consumers before touching Metal. Opaque MPS
  # resources are additionally bounded by the external process guard.
  scratch_bytes = ML::Video::HyNomasDualStreamBlock.scratch_plan(config).sum { |_, count| count.to_i64 * 4 }
  refiner_bytes = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(base.mask, retain_padding: true).total
  attention_bytes = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(
    1, config.video_tokens + base.valid_text_tokens, 16, 128, 256, 16).resident_bytes
  # The remaining plans have under 768 MiB of named stages at this fixed
  # geometry (linear weight stage, RMS, patch, inputs, and compact text).
  preflight = globals_plan.destination_bytes + block_plan.destination_bytes + scratch_bytes +
              refiner_bytes + attention_bytes + 768_i64 * 1024**2
  raise ArgumentError.new("real block preflight exceeds named budget: #{preflight}") unless preflight <= budget
  HyNomasGeneration.emit("real_block_admitted", preflight_named_upper_bound: preflight,
    valid_text_tokens: base.valid_text_tokens, storage_text_tokens: config.text_tokens)
  input_arena = ML::Video::Arena.new
  arenas << input_arena
  globals = ML::Video::Arena.new
  arenas << globals
  weights = ML::Video::Arena.new
  arenas << weights
  scratch = ML::Video::Arena.new
  arenas << scratch
  globals_prepared = ML::Video::PreparedWeights.new(globals_plan, globals)
  prepared << globals_prepared
  block_prepared = ML::Video::PreparedWeights.new(block_plan, weights)
  prepared << block_prepared
  timeline = inputs.upload!("input.latent_bank", input_arena)
  qwen = base.upload!("input.qwen_bf16", input_arena)
  cos = base.upload!("input.rope_cos", input_arena)
  sin = base.upload!("input.rope_sin", input_arena)
  latent = input_arena.alloc("segment.latent", 32 * 30 * 9 * 16)
  velocity = input_arena.alloc("unused.velocity", 32 * 30 * 9 * 16)
  txt = input_arena.alloc("compact.txt", config.text_tokens * 2048)
  prefix = ML::Video::HyNomasTimelineOps::PrefixPlan.new(
    ML::Video::HyNomasTimelineOps::PrefixGeometry.new(channels: 32, total_frames: 180,
      prefix_frames: 30, height: 9, width: 16), source: timeline, destination: latent)
  patch = ML::Video::HyNomasPatchMPSPlan.new(latent, 4320, globals)
  refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(qwen, base.mask, 1000_f32, globals,
    refiner_bytes, retain_padding: true)
  euler = ML::Video::HyNomasEulerPlan.new(latent, velocity,
    refiner.checkpoints["timestep"], steps: 4, shift: 5_f32)
  ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch, config)
  block = ML::Video::HyNomasDualStreamBlock.new(config, weights, scratch)
  workspace = ML::Video::HyNomasMPSWorkspace.new(block, patch.output, txt,
    refiner.vec, refiner.vec_txt, cos, sin, attention_query_chunk: 256,
    attention_heads_per_group: 16, valid_text_tokens: base.valid_text_tokens,
    text_rms_tokens: 1280, allow_shared_modulation: true)
  named = arenas.sum(&.resident_bytes) + patch.resident_bytes + refiner.resident_bytes +
          workspace.resident_bytes + euler.resident_bytes
  raise "actual named buffers exceeded preflight/budget" unless named <= preflight && named <= budget
  captures = {
    "input.latent" => latent, "input.img" => patch.output, "input.txt" => txt,
    "input.vec" => refiner.vec, "input.rope_cos" => cos, "input.rope_sin" => sin,
  }
  refiner.checkpoints.each { |name, buffer| captures["conditioning.#{name}"] = buffer }
  block.checkpoints.each { |name, buffer| captures["block.#{name}"] = buffer }
  chain = ML::Metal::MPSCommandChain.new
  before_allocations = ML::MetalBuffer.hy_text_probe_allocations
  before_reads = ML::MetalBuffer.hy_text_probe_readbacks
  started = Time.instant
  globals_prepared.encode(chain.encoder)
  block_prepared.encode(chain.encoder)
  prefix.encode!(chain)
  euler.encode_timestep(chain, 0)
  patch.encode(chain)
  refiner.encode(chain)
  ML::Video::MetalOps.copy(chain.encoder, refiner.text_output, txt, config.text_tokens * 2048)
  block.encode_mps(chain, patch.output, txt, refiner.vec, refiner.vec_txt, cos, sin, workspace)
  encode_seconds = (Time.instant - started).total_seconds
  allocations = ML::MetalBuffer.hy_text_probe_allocations - before_allocations
  reads = ML::MetalBuffer.hy_text_probe_readbacks - before_reads
  raise "real block encode allocated/read back/waited" unless allocations == 0 && reads == 0 && chain.wait_count == 0
  chain.commit_and_wait
  raise "real block diagnostic requires one terminal wait" unless chain.wait_count == 1
  records = {} of String => NamedTuple(dtype: String, shape: Array(Int64), offset: Int64, nbytes: Int64, sha256: String)
  offset = 0_i64
  HyNomasGeneration.write_fresh_output(payload) do |io|
    captures.each do |name, buffer|
      # Borrow shared storage after the terminal wait; never copy entire
      # captures into a second CPU tensor or serialize numeric JSON arrays.
      bytes = Bytes.new(buffer.contents.as(UInt8*), buffer.size.to_i32)
      io.write(bytes)
      records[name] = {dtype: "F32", shape: [buffer.size // 4], offset: offset,
                       nbytes: buffer.size, sha256: Digest::SHA256.hexdigest(bytes)}
      offset += buffer.size
    end
  end
  payload_sha = PixelUMMBlockProbe.sha_file(payload)
  HyNomasGeneration.write_fresh_output(output) do |io|
    JSON.build(io, indent: "  ") do |json|
      json.object do
        json.field "schema", "cogni_ml.hy_nomas_real_block_capture/v1"
        json.field "conditioning_manifest", File.realpath(manifest)
        json.field "conditioning_sha256", manifest_sha
        json.field "weight_index_sha256", HyNomasGeneration::INDEX_SHA
        json.field "cache_index", File.realpath(HyNomasGeneration::INDEX)
        json.field "cache_index_sha256", HyNomasGeneration::INDEX_SHA
        json.field "block_weights", pin.bf16_path
        json.field "block_weights_sha256", pin.bf16_sha256
        json.field "global_weights", index.globals_bf16_path
        json.field "global_weights_sha256", index.globals_bf16_sha256
        json.field "geometry", {batches: 1, video_tokens: 4320, text_tokens: config.text_tokens,
                                valid_text_tokens: base.valid_text_tokens, heads: 16, head_dim: 128, dim: 2048,
                                ffn_dim: 8192, text_rms_tokens: 1280}
        json.field "timestep", 1000
        json.field "scope", "first segment, first step, patch/refiner/block0; no history, full DiT or media generation"
        json.field "precision", "BF16 resident weights; F32 activations and MPS attention control"
        json.field "named_bytes", named
        json.field "named_allocations_during_encode", allocations
        json.field "activation_readbacks_before_terminal_wait", reads
        json.field "terminal_wait_count", chain.wait_count
        json.field "encode_seconds", encode_seconds
        json.field "gpu_elapsed_seconds", chain.gpu_elapsed_seconds
        json.field "payload", File.basename(payload)
        json.field "payload_bytes", offset
        json.field "payload_sha256", payload_sha
        json.field "tensors", records
      end
    end
  end
  HyNomasGeneration.emit("real_block_captured", capture: output, sha256: PixelUMMBlockProbe.sha_file(output),
    named_bytes: named, tensor_count: records.size)
ensure
  Qwen25PrefixProbe.cleanup_after_drain(chain) do
    workspace.try(&.release)
    euler.try(&.release)
    refiner.try(&.release)
    patch.try(&.release)
    if active = chain
      prefix.try(&.release_after_drain!(active))
    end
    chain.try(&.release)
    prepared.reverse_each(&.release)
    arenas.reverse_each(&.release)
    files.reverse_each(&.close)
  end
end
