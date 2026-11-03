require "option_parser"
require "json"
require "../src/ml/video/hy_nomas_timeline_ops"
require "../src/ml/video/hy_nomas_text_refiner"
require "../src/ml/video/hy_nomas_dit_boundaries"
require "./support/hy_nomas_generation"
require "./support/hy_nomas_timeline_inputs"
require "./support/hy_nomas_reference_latent"
require "./support/hy_nomas_reference_cache_probe"
require "./support/hy_nomas_reference_refiner_trace"
require "./support/hy_nomas_probe_instrumentation"

module HyNomasReferenceCacheProbeCLI
  extend self

  MAX_NAMED_BYTES = 8_i64 * 1024**3
  TOKENS          = 4320
  DIM             = 2048
  CHANNELS        =   32

  def patch_named_bytes : Int64
    (TOKENS.to_i64 * CHANNELS + TOKENS.to_i64 * DIM + DIM.to_i64 * CHANNELS) * sizeof(Float32)
  end

  def bindings : Array(ML::Video::WeightBinding)
    HyNomasQ4DiTIO::GLOBAL_BINDINGS.map do |binding|
      ML::Video::WeightBinding.new(binding.target, binding.source, binding.shape,
        ML::GGUF::SafeTensorDType::BF16)
    end
  end
end

timeline_path = ""
timeline_sha = ""
reference_latent_path = ""
reference_latent_sha = ""
output_manifest = ""
refiner_trace_output_manifest = ""
prelude_input_bf16_control = false
max_named_bytes = 4_i64 * 1024**3
execute = false
OptionParser.parse do |parser|
  parser.banner = "Export native reference-T0 timeline/noise/patch/refiner inputs for standalone block-0 cache comparison."
  parser.on("--timeline-inputs PATH", "Authenticated six-segment timeline manifest") { |value| timeline_path = value }
  parser.on("--timeline-inputs-sha256 SHA", "Trusted timeline manifest SHA-256") { |value| timeline_sha = value }
  parser.on("--reference-latent PATH", "Authenticated saved BF16 DCAE latent safetensors") { |value| reference_latent_path = value }
  parser.on("--reference-latent-sha256 SHA", "Trusted saved latent artifact SHA-256") { |value| reference_latent_sha = value }
  parser.on("--output PATH", "Fresh output JSON manifest; sidecar is a sibling .tensors.bin") { |value| output_manifest = value }
  parser.on("--refiner-trace-output PATH", "Optional fresh separate refiner trace JSON manifest; sidecar is a sibling .tensors.bin") { |value| refiner_trace_output_manifest = value }
  parser.on("--refiner-input-bf16-control", "Round only the native refiner mean and timestep sinusoidal inputs through BF16") { prelude_input_bf16_control = true }
  parser.on("--max-named-bytes N", "Aggregate named Metal-buffer cap, up to 8 GiB") { |value| max_named_bytes = value.to_i64 }
  parser.on("--execute", "Run native timeline/patch/refiner on MPS and capture after one terminal drain") { execute = true }
end

unless ARGV.empty? && !timeline_path.empty? && !timeline_sha.empty? &&
       !reference_latent_path.empty? && !reference_latent_sha.empty? &&
       max_named_bytes > 0 && max_named_bytes <= HyNomasReferenceCacheProbeCLI::MAX_NAMED_BYTES
  raise ArgumentError.new("timeline and reference-latent identities plus a positive named-buffer cap are required")
end
raise ArgumentError.new("--execute requires a fresh output manifest path") if execute && output_manifest.empty?
if !refiner_trace_output_manifest.empty? && !execute
  raise ArgumentError.new("--refiner-trace-output requires --execute")
end
if prelude_input_bf16_control && refiner_trace_output_manifest.empty?
  raise ArgumentError.new("--refiner-input-bf16-control requires --refiner-trace-output")
end

timeline_inputs = HyNomasGeneration::TimelineInputs.new(timeline_path, timeline_sha)
reference_input = HyNomasReferenceLatent::Input.new(reference_latent_path, reference_latent_sha)
timeline_inputs.verify_payload!
timeline_inputs.base.file.verify_payload!
valid_text_tokens = timeline_inputs.base.valid_text_tokens
unless 0 < valid_text_tokens <= 1280
  raise ArgumentError.new("reference cache probe requires a nonempty admitted text prefix")
end

# Authenticate the exact production weight/cache index even for plan-only
# admission. No Metal model is prepared unless --execute is explicit.
admission = HyNomasGeneration.authenticate_bf16_index
global_file = ML::Video::BF16WeightCache.authenticate(admission.globals_bf16_path,
  admission.globals_bf16_sha256)
weight_plan = ML::Video::WeightPlan.new([global_file], HyNomasReferenceCacheProbeCLI.bindings,
  max_named_bytes, resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
unless weight_plan.staging_bytes == 0 && weight_plan.entries.size == HyNomasQ4DiTIO::GLOBAL_BINDINGS.size
  global_file.close
  reference_input.close
  raise ArgumentError.new("reference cache global weights must use exact direct BF16 residents")
end

mask = timeline_inputs.base.mask
text_named_bytes = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(mask, retain_padding: true).total
base_input_bytes = timeline_inputs.base.bytes(%w(input.qwen_bf16 input.rope_cos input.rope_sin))
timeline_input_bytes = timeline_inputs.bytes(%w(input.latent_bank input.rope_cos input.rope_sin cache.eps_bf16))
geometry = ML::Video::HyNomasTimelineOps::Geometry.new
segment_bytes = geometry.segment_bytes
reference_bytes = HyNomasReferenceLatent::BF16_BYTES
planned_named_bytes = weight_plan.peak_bytes + base_input_bytes + timeline_input_bytes +
                      segment_bytes + reference_bytes +
                      HyNomasReferenceCacheProbeCLI.patch_named_bytes + text_named_bytes
unless planned_named_bytes <= max_named_bytes
  global_file.close
  reference_input.close
  raise ArgumentError.new("reference cache probe exceeds its named-buffer cap")
end

plan_report = {
  schema:                           "cogni_ml.hy_nomas_reference_cache_probe_plan/v1",
  scope:                            HyNomasReferenceCacheProbe::SCOPE,
  timeline_manifest_sha256:         timeline_inputs.file.sha256,
  reference_latent_artifact_sha256: reference_input.sha256,
  valid_text_tokens:                timeline_inputs.base.valid_text_tokens,
  tensor_names:                     HyNomasReferenceCacheProbe::TENSOR_NAMES,
  named_bytes:                      planned_named_bytes,
  max_named_bytes:                  max_named_bytes,
  mps_execution:                    execute,
  upstream_preprocessing_parity:    false,
}.to_pretty_json
unless execute
  puts plan_report
  global_file.close
  reference_input.close
  exit 0
end

manifest_path, payload_path = HyNomasReferenceCacheProbe.checked_output_paths!(output_manifest)
HyNomasReferenceCacheProbe.ensure_fresh_outputs!(manifest_path, payload_path)
refiner_trace_paths : Tuple(String, String)? = nil
unless refiner_trace_output_manifest.empty?
  trace_manifest_path, trace_payload_path = HyNomasReferenceRefinerTrace.checked_output_paths!(refiner_trace_output_manifest)
  HyNomasReferenceRefinerTrace.ensure_fresh_outputs!(trace_manifest_path, trace_payload_path)
  HyNomasReferenceRefinerTrace.ensure_distinct_outputs!(manifest_path, payload_path,
    trace_manifest_path, trace_payload_path)
  refiner_trace_paths = {trace_manifest_path, trace_payload_path}
end

base_arena = ML::Video::Arena.new
timeline_arena = ML::Video::Arena.new
working_arena = ML::Video::Arena.new
global_arena = ML::Video::Arena.new
reference_latent : ML::MetalBuffer? = nil
prepared : ML::Video::PreparedWeights? = nil
patch : ML::Video::HyNomasPatchMPSPlan? = nil
refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
timeline_plan : ML::Video::HyNomasTimelineOps::Plan? = nil
chain : ML::Metal::MPSCommandChain? = nil
wait_attempted = false

begin
  base = timeline_inputs.base
  qwen = base.upload!("input.qwen_bf16", base_arena)
  cosine = base.upload!("input.rope_cos", base_arena)
  sine = base.upload!("input.rope_sin", base_arena)
  latent_timeline = timeline_inputs.upload!("input.latent_bank", timeline_arena)
  cosine_bank = timeline_inputs.upload!("input.rope_cos", timeline_arena)
  sine_bank = timeline_inputs.upload!("input.rope_sin", timeline_arena)
  cache_noise = timeline_inputs.upload!("cache.eps_bf16", timeline_arena)
  segment = working_arena.alloc("timeline.segment_latent", geometry.segment_elements)
  reference_latent = reference_input.upload_bf16!

  prepared = ML::Video::PreparedWeights.new(weight_plan, global_arena)
  patch = ML::Video::HyNomasPatchMPSPlan.new(segment, HyNomasReferenceCacheProbeCLI::TOKENS, global_arena)
  refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(qwen, base.mask, 1000_f32,
    global_arena, text_named_bytes, retain_padding: true,
    prelude_input_bf16_control: prelude_input_bf16_control)
  timeline_plan = ML::Video::HyNomasTimelineOps::Plan.new(geometry,
    timeline: latent_timeline, segment_latent: segment,
    cosine_bank: cosine_bank, sine_bank: sine_bank,
    cosine: cosine, sine: sine, cache_eps_bf16: cache_noise,
    cache_timestep: refiner.checkpoints["timestep"], run_segments: 2,
    reference_initialization: true)

  actual_named_bytes = base_arena.resident_bytes + timeline_arena.resident_bytes +
                       working_arena.resident_bytes + global_arena.resident_bytes +
                       reference_latent.size + patch.resident_bytes + refiner.resident_bytes
  unless actual_named_bytes == planned_named_bytes && actual_named_bytes <= max_named_bytes
    raise ArgumentError.new("reference cache probe allocation/admission mismatch: #{actual_named_bytes} != #{planned_named_bytes}")
  end

  chain = ML::Metal::MPSCommandChain.new
  prepared.encode(chain.not_nil!.encoder)
  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
  timeline_plan.admit_reference_latent!(chain.not_nil!, reference_latent)
  timeline_plan.prepare_reference_segment!(chain.not_nil!, 0)
  timeline_plan.write_cache_noise!(chain.not_nil!, 0)
  patch.encode(chain.not_nil!)
  refiner.encode(chain.not_nil!)
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  preterminal_waits = chain.not_nil!.wait_count
  unless encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0
    raise "reference preblock chain allocated, read back, or waited before terminal drain"
  end

  wait_attempted = true
  chain.not_nil!.commit_and_wait
  unless chain.not_nil!.drained? && chain.not_nil!.wait_count == 1
    raise "reference preblock export requires exactly one terminal drain"
  end

  text_prefix_bytes = valid_text_tokens * HyNomasReferenceCacheProbeCLI::DIM * sizeof(Float32)
  sources = [
    HyNomasReferenceCacheProbe::TensorSource.new("reference.latent_bf16", reference_latent.not_nil!, "BF16", [1, 32, 30, 9, 16], HyNomasReferenceLatent::BF16_BYTES.to_i32),
    HyNomasReferenceCacheProbe::TensorSource.new("cache.noisy_latent_f32", segment, "F32", [1, 32, 30, 9, 16], geometry.segment_bytes.to_i32),
    HyNomasReferenceCacheProbe::TensorSource.new("block.img_f32", patch.not_nil!.output, "F32", [1, HyNomasReferenceCacheProbeCLI::TOKENS, HyNomasReferenceCacheProbeCLI::DIM], HyNomasReferenceCacheProbeCLI::TOKENS * HyNomasReferenceCacheProbeCLI::DIM * sizeof(Float32)),
    HyNomasReferenceCacheProbe::TensorSource.new("block.txt_valid_f32", refiner.not_nil!.text_output, "F32", [1, valid_text_tokens, HyNomasReferenceCacheProbeCLI::DIM], text_prefix_bytes),
    HyNomasReferenceCacheProbe::TensorSource.new("block.vec_f32", refiner.not_nil!.vec, "F32", [1, HyNomasReferenceCacheProbeCLI::DIM], HyNomasReferenceCacheProbeCLI::DIM * sizeof(Float32)),
    HyNomasReferenceCacheProbe::TensorSource.new("block.vec_txt_f32", refiner.not_nil!.vec_txt, "F32", [1, HyNomasReferenceCacheProbeCLI::DIM], HyNomasReferenceCacheProbeCLI::DIM * sizeof(Float32)),
    HyNomasReferenceCacheProbe::TensorSource.new("block.rope_cos_f32", cosine, "F32", [HyNomasReferenceCacheProbeCLI::TOKENS, 128], HyNomasReferenceCacheProbeCLI::TOKENS * 128 * sizeof(Float32)),
    HyNomasReferenceCacheProbe::TensorSource.new("block.rope_sin_f32", sine, "F32", [HyNomasReferenceCacheProbeCLI::TOKENS, 128], HyNomasReferenceCacheProbeCLI::TOKENS * 128 * sizeof(Float32)),
    HyNomasReferenceCacheProbe::TensorSource.new("cache.timestep_f32", refiner.not_nil!.checkpoints["timestep"], "F32", [1], sizeof(Float32)),
  ]
  result = HyNomasReferenceCacheProbe.write_fixture!(manifest_path,
    reference_latent_artifact_sha256: reference_input.sha256,
    timeline_manifest_sha256: timeline_inputs.file.sha256,
    valid_text_tokens: valid_text_tokens,
    sources: sources)
  if trace_paths = refiner_trace_paths
    trace_sources = HyNomasReferenceRefinerTrace.tensor_sources(refiner.not_nil!.checkpoints,
      valid_text_tokens)
    trace_result = HyNomasReferenceRefinerTrace.write_trace!(trace_paths[0],
      reference_latent_artifact_sha256: reference_input.sha256,
      timeline_manifest_sha256: timeline_inputs.file.sha256,
      cache_fixture_manifest_sha256: result["manifest_sha256"].as_s,
      valid_text_tokens: valid_text_tokens,
      sources: trace_sources,
      encode_allocations: encode_allocations,
      encode_readbacks: encode_readbacks,
      preterminal_waits: preterminal_waits,
      terminal_waits: chain.not_nil!.wait_count,
      prelude_input_bf16_control: prelude_input_bf16_control)
    puts trace_result.to_json
  end
  puts plan_report
  puts result.to_json
  puts({schema: "cogni_ml.hy_nomas_reference_cache_probe_result/v1",
        scope: HyNomasReferenceCacheProbe::SCOPE,
        native_cache_writer_executed: false,
        exported_for_standalone_block0_cache_writer: true,
        encode_allocations: encode_allocations, encode_readbacks: encode_readbacks,
        preterminal_waits: preterminal_waits, terminal_waits: chain.not_nil!.wait_count,
        actual_named_bytes: actual_named_bytes, passed: true}.to_pretty_json)
ensure
  if active_chain = chain
    unless wait_attempted || active_chain.drained?
      wait_attempted = true
      begin
        active_chain.commit_and_wait
      rescue
      end
    end
    if active_chain.drained?
      timeline_plan.try(&.release_after_drain!(active_chain))
      refiner.try(&.release)
      patch.try(&.release)
      prepared.try(&.release)
      active_chain.release
      base_arena.release
      timeline_arena.release
      working_arena.release
      global_arena.release
      reference_latent.try(&.release)
      global_file.close
      reference_input.close
    else
      HyNomasReferenceCacheProbe.retain_in_flight!(-> {
        active_chain.object_id
        timeline_inputs.object_id
        reference_input.object_id
        reference_latent.try(&.object_id)
        timeline_plan.try(&.object_id)
        refiner.try(&.object_id)
        patch.try(&.object_id)
        prepared.try(&.object_id)
        base_arena.object_id
        timeline_arena.object_id
        working_arena.object_id
        global_arena.object_id
        global_file.object_id
        nil
      })
    end
  else
    refiner.try(&.release)
    patch.try(&.release)
    prepared.try(&.release)
    base_arena.release
    timeline_arena.release
    working_arena.release
    global_arena.release
    reference_latent.try(&.release)
    global_file.close
    reference_input.close
  end
end
