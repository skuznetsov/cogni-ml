require "option_parser"
require "json"
require "digest/sha256"
require "./support/hy_nomas_generation"
require "./support/qwen25_prefix_probe"
require "./support/hy_nomas_probe_instrumentation"
require "../src/ml/video/hy_nomas_dit_boundaries"
require "../src/ml/video/hy_nomas_mps_bf16_linear"

# Diagnostic-only BF16 patch projection. This uploads the authenticated real
# first-step latent, rounds it to BF16 on GPU, applies the production T/H/W
# token transpose and the generic physical-BF16 linear plan, then captures
# bytes after the sole terminal wait. It is not a production backend.
module HyNomasBF16PatchProbe
  extend self

  SCHEMA            = "cogni_ml.hy_nomas_bf16_patch_case/v1"
  OUTPUT_SCHEMA     = "cogni_ml.hy_nomas_bf16_patch_capture/v1"
  PINNED_REVISION   = "911cea6458a89c2663513ff7099561ec190dd491"
  CHANNELS          =   32
  DIM               = 2048
  FRAMES            =   30
  HEIGHT            =    9
  WIDTH             =   16
  ROWS              = FRAMES * HEIGHT * WIDTH
  INPUT_ELEMENTS    = CHANNELS * ROWS
  OUTPUT_ELEMENTS   = ROWS * DIM
  INPUT_BYTES       = INPUT_ELEMENTS.to_i64 * sizeof(Float32)
  OUTPUT_BF16_BYTES = OUTPUT_ELEMENTS.to_i64 * 2
  OUTPUT_F32_BYTES  = OUTPUT_ELEMENTS.to_i64 * sizeof(Float32)
  MAX_NAMED_BYTES   = 128_i64 * 1024**2
  OPERATION         = "pinned PatchEmbed.forward with BF16 input and resident BF16 pointwise Conv3d weights; flatten/transpose [T,H,W]"

  RNE_SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    inline uint hy_nomas_patch_bf16_rne(uint bits) {
      const uint low = bits & 0xffffu;
      const uint high = bits >> 16;
      const uint lsb = high & 1u;
      return ((bits + 0x7fffu + lsb) >> 16) << 16;
    }

    kernel void hy_nomas_bf16_patch_rne_inplace_f32(
        device float *values [[buffer(0)]],
        constant uint &count [[buffer(1)]],
        uint index [[thread_position_in_grid]]) {
      if (index < count) {
        values[index] = as_type<float>(hy_nomas_patch_bf16_rne(as_type<uint>(values[index])));
      }
    }
  METAL

  def valid_digest?(value : String) : Bool
    value.size == 64 && value.each_char.all? { |char| ('0'..'9').includes?(char) || ('a'..'f').includes?(char) }
  end

  def sha256(path : String) : String
    digest = Digest::SHA256.new
    File.open(path) do |io|
      scratch = Bytes.new(1024 * 1024)
      while count = io.read(scratch)
        break if count == 0
        digest.update(scratch[0, count])
      end
    end
    digest.hexfinal
  end

  def canonical_file(path : String, label : String) : String
    unless Path[path].absolute? && File.file?(path) && !File.symlink?(path) && File.realpath(path) == path
      raise ArgumentError.new("#{label} must be a canonical absolute regular file")
    end
    path
  end

  def read_record!(doc : JSON::Any, payload_path : String, name : String,
                   dtype : String, shape : Array(Int64), offset : Int64,
                   nbytes : Int64, payload_bytes : Int64) : Bytes
    record = doc["tensors"][name]
    unless record["dtype"].as_s == dtype &&
           record["shape"].as_a.map(&.as_i64) == shape &&
           record["offset"].as_i64 == offset && record["nbytes"].as_i64 == nbytes &&
           offset >= 0 && nbytes > 0 && offset <= payload_bytes &&
           nbytes <= payload_bytes - offset && valid_digest?(record["sha256"].as_s)
      raise ArgumentError.new("BF16 patch case tensor descriptor mismatch: #{name}")
    end
    bytes = Bytes.new(nbytes.to_i32)
    File.open(payload_path) do |io|
      io.seek(offset)
      io.read_fully(bytes)
    end
    unless Digest::SHA256.hexdigest(bytes) == record["sha256"].as_s
      raise ArgumentError.new("BF16 patch case tensor digest mismatch: #{name}")
    end
    bytes
  end

  def parse_case!(path : String, supplied_sha : String) : Tuple(JSON::Any, String, Bytes)
    canonical_file(path, "patch case manifest")
    unless valid_digest?(supplied_sha) && sha256(path) == supplied_sha
      raise ArgumentError.new("BF16 patch case manifest digest mismatch")
    end
    doc = JSON.parse(File.read(path))
    unless doc["schema"].as_s == SCHEMA &&
           doc["upstream_revision"].as_s == PINNED_REVISION &&
           doc["operation"].as_s == OPERATION &&
           doc["torch_device"].as_s == "mps" &&
           doc["torch_input_dtype"].as_s == "bfloat16" &&
           doc["torch_output_dtype"].as_s == "bfloat16" &&
           doc["weight_index_sha256"].as_s == HyNomasGeneration::INDEX_SHA &&
           doc["geometry"]["input_shape"].as_a.map(&.as_i64) == [1_i64, CHANNELS.to_i64, FRAMES.to_i64, HEIGHT.to_i64, WIDTH.to_i64] &&
           doc["geometry"]["rows"].as_i == ROWS.to_i64 &&
           doc["geometry"]["input_width"].as_i == CHANNELS.to_i64 &&
           doc["geometry"]["output_width"].as_i == DIM.to_i64 &&
           doc["weights"]["weight_name"].as_s == "img_in.proj.weight" &&
           doc["weights"]["bias_name"].as_s == "img_in.proj.bias" &&
           doc["weights"]["weight_shape"].as_a.map(&.as_i64) == [DIM.to_i64, CHANNELS.to_i64, 1_i64, 1_i64, 1_i64] &&
           doc["weights"]["bias_shape"].as_a.map(&.as_i64) == [DIM.to_i64] &&
           valid_digest?(doc["capture_manifest_sha256"].as_s) &&
           valid_digest?(doc["capture_payload_sha256"].as_s) &&
           valid_digest?(doc["capture_latent_sha256"].as_s) &&
           valid_digest?(doc["global_weights_sha256"].as_s)
      raise ArgumentError.new("BF16 patch case source or geometry contract mismatch")
    end
    payload_name = doc["payload"].as_s
    unless Path[payload_name].basename == payload_name && payload_name.ends_with?(".tensors.bin")
      raise ArgumentError.new("BF16 patch case payload must be a sibling tensor file")
    end
    payload_path = File.join(File.dirname(path), payload_name)
    canonical_file(payload_path, "patch case payload")
    payload_bytes = doc["payload_bytes"].as_i64
    payload_sha = doc["payload_sha256"].as_s
    unless payload_bytes == INPUT_BYTES + OUTPUT_BF16_BYTES &&
           valid_digest?(payload_sha) && File.size(payload_path) == payload_bytes &&
           sha256(payload_path) == payload_sha
      raise ArgumentError.new("BF16 patch case payload extent or digest mismatch")
    end
    records = doc["tensors"]
    unless records.as_h.keys.sort == ["input.latent", "patch.expected_bf16"]
      raise ArgumentError.new("BF16 patch case tensor inventory mismatch")
    end
    input = read_record!(doc, payload_path, "input.latent", "F32",
      [1_i64, CHANNELS.to_i64, FRAMES.to_i64, HEIGHT.to_i64, WIDTH.to_i64], 0_i64, INPUT_BYTES, payload_bytes)
    unless doc["capture_latent_sha256"].as_s == doc["tensors"]["input.latent"]["sha256"].as_s
      raise ArgumentError.new("BF16 patch case latent is not bound to the authenticated capture latent")
    end
    read_record!(doc, payload_path, "patch.expected_bf16", "BF16",
      [1_i64, ROWS.to_i64, DIM.to_i64], INPUT_BYTES, OUTPUT_BF16_BYTES, payload_bytes)
    {doc, payload_path, input}
  end
end

manifest = ""
manifest_sha = ""
output = ""
variant_name = ""
budget = 0_i64
OptionParser.parse do |parser|
  parser.banner = "One actual BF16 img_in.proj patch operation on the authenticated first-step latent."
  parser.on("--case PATH", "Torch-MPS BF16 patch case manifest") { |v| manifest = v }
  parser.on("--case-sha256 SHA", "Trusted case manifest SHA-256") { |v| manifest_sha = v }
  parser.on("--output PATH", "New native capture JSON and sibling tensor sidecar") { |v| output = v }
  parser.on("--variant NAME", "bf16-graph or f32-accum") { |v| variant_name = v }
  parser.on("--max-named-bytes N", "Aggregate named Metal buffer budget (at most 128 MiB)") { |v| budget = v.to_i64 }
end
unless ARGV.empty? && !manifest.empty? && !manifest_sha.empty? && !output.empty? &&
       output.ends_with?(".json") && !variant_name.empty? &&
       0 < budget <= HyNomasBF16PatchProbe::MAX_NAMED_BYTES
  raise ArgumentError.new("case, digest, fresh .json output, explicit graph variant and a budget up to 128 MiB are required")
end
variant = ML::Video::HyNomasMPSBF16Linear::Variant.parse(variant_name)
required_named_bytes = HyNomasBF16PatchProbe::INPUT_BYTES +
                       HyNomasBF16PatchProbe::ROWS.to_i64 * HyNomasBF16PatchProbe::CHANNELS * 4 +
                       HyNomasBF16PatchProbe::ROWS.to_i64 * HyNomasBF16PatchProbe::DIM * 2 +
                       HyNomasBF16PatchProbe::ROWS.to_i64 * HyNomasBF16PatchProbe::DIM * 4 +
                       (HyNomasBF16PatchProbe::DIM.to_i64 * HyNomasBF16PatchProbe::CHANNELS + HyNomasBF16PatchProbe::DIM) * 2
raise ArgumentError.new("BF16 patch named buffers exceed the supplied budget") if required_named_bytes > budget
payload = output.sub(/\.json\z/, ".tensors.bin")
{output, payload}.each do |path|
  raise ArgumentError.new("refusing to replace output #{path}") if File.exists?(path) || File.symlink?(path)
end

case_doc, _case_payload_path, case_input = HyNomasBF16PatchProbe.parse_case!(manifest, manifest_sha)
index = HyNomasGeneration.authenticate_bf16_index
unless index.sha256 == case_doc["weight_index_sha256"].as_s &&
       index.globals_bf16_sha256 == case_doc["global_weights_sha256"].as_s
  raise ArgumentError.new("BF16 patch case and native global checkpoint identities differ")
end

bindings = ML::Video::HyNomasPatchMPSPlan.checkpoint_bindings.map do |binding|
  diagnostic_target = case binding.target
                      when "img_in.proj.weight" then "img_in.proj_weight"
                      when "img_in.proj.bias"   then "img_in.proj_bias"
                      else
                        raise ArgumentError.new("unexpected production patch weight target: #{binding.target}")
                      end
  ML::Video::WeightBinding.new(diagnostic_target, binding.source, binding.shape,
    ML::GGUF::SafeTensorDType::BF16)
end
unless bindings.size == 2 &&
       bindings.map(&.target).sort == ["img_in.proj_bias", "img_in.proj_weight"] &&
       bindings.map(&.source).sort == ["img_in.proj.bias", "img_in.proj.weight"] &&
       bindings.find { |binding| binding.source == "img_in.proj.weight" }.not_nil!.shape ==
         [HyNomasBF16PatchProbe::DIM.to_i64, HyNomasBF16PatchProbe::CHANNELS.to_i64, 1_i64, 1_i64, 1_i64] &&
       bindings.find { |binding| binding.source == "img_in.proj.bias" }.not_nil!.shape == [HyNomasBF16PatchProbe::DIM.to_i64]
  raise ArgumentError.new("production pointwise patch checkpoint binding changed")
end

input_arena = ML::Video::Arena.new
weight_arena = ML::Video::Arena.new
output_arena = ML::Video::Arena.new
files = [] of ML::Video::CheckpointFile
prepared : ML::Video::PreparedWeights? = nil
plan : ML::Video::HyNomasMPSBF16Linear::Plan? = nil
chain : ML::Metal::MPSCommandChain? = nil
begin
  global_file = ML::Video::BF16WeightCache.authenticate(index.globals_bf16_path, index.globals_bf16_sha256)
  files << global_file
  weight_plan = ML::Video::WeightPlan.new([global_file], bindings, budget,
    resident_dtype: ML::Video::WeightDType::BF16, require_all: false)
  expected_weight_bytes = (HyNomasBF16PatchProbe::DIM.to_i64 * HyNomasBF16PatchProbe::CHANNELS +
                           HyNomasBF16PatchProbe::DIM) * 2
  unless weight_plan.entries.size == 2 && weight_plan.destination_bytes == expected_weight_bytes &&
         weight_plan.staging_bytes == 0 && !weight_plan.allow_f32_to_bf16
    raise ArgumentError.new("BF16 patch weights are not an exact direct-copy pair")
  end
  weights_prepared = ML::Video::PreparedWeights.new(weight_plan, weight_arena)
  prepared = weights_prepared
  linear_weights = ML::Video::HyNomasDualStreamBlock::LinearWeights.new(weight_arena, "img_in.proj")

  input = input_arena.alloc("patch.input_f32", HyNomasBF16PatchProbe::INPUT_ELEMENTS)
  input.write_bytes(case_input.to_unsafe, case_input.size)
  tokens = input_arena.alloc("patch.tokens_f32", HyNomasBF16PatchProbe::ROWS * HyNomasBF16PatchProbe::CHANNELS)
  output_bf16 = output_arena.alloc_weight("patch.output_bf16", HyNomasBF16PatchProbe::OUTPUT_ELEMENTS,
    ML::Video::WeightDType::BF16)
  output_f32 = output_arena.alloc("patch.output_f32", HyNomasBF16PatchProbe::OUTPUT_ELEMENTS)
  named_bytes = input_arena.resident_bytes + weight_arena.resident_bytes + output_arena.resident_bytes
  unless named_bytes == required_named_bytes && named_bytes <= budget
    raise ArgumentError.new("BF16 patch named resident bytes exceed the supplied budget")
  end

  linear_plan = ML::Video::HyNomasMPSBF16Linear::Plan.new(
    tokens, linear_weights, output_bf16, output_f32, variant,
    HyNomasBF16PatchProbe::ROWS, HyNomasBF16PatchProbe::CHANNELS,
    HyNomasBF16PatchProbe::DIM)
  plan = linear_plan
  rne = ML::Metal::ComputePipeline.new(
    "hy_nomas_bf16_patch_rne_inplace_f32", HyNomasBF16PatchProbe::RNE_SOURCE)
  transpose = ML::Metal::PipelineCache.get("hy_nomas_dit_channels_to_tokens_f32") do
    ML::Metal::ComputePipeline.new("hy_nomas_dit_channels_to_tokens_f32",
      ML::Video::HyNomasPatchMPSPlan::SOURCE)
  end
  chain = ML::Metal::MPSCommandChain.new
  weights_prepared.encode(chain.encoder)
  before_allocations = ML::MetalBuffer.hy_text_probe_allocations
  before_reads = ML::MetalBuffer.hy_text_probe_readbacks
  before_waits = chain.wait_count
  encoder = chain.encoder
  encoder.set_pipeline(rne).set_buffer(input, 0, ML::Metal::BufferAccess::ReadWrite)
    .set_value(HyNomasBF16PatchProbe::INPUT_ELEMENTS.to_u32, 1)
    .dispatch_1d(HyNomasBF16PatchProbe::INPUT_ELEMENTS)
  encoder.set_pipeline(transpose).set_buffer(input, 0)
    .set_buffer(tokens, 1, ML::Metal::BufferAccess::Write)
    .set_value(HyNomasBF16PatchProbe::ROWS.to_u32, 2)
    .set_value(HyNomasBF16PatchProbe::CHANNELS.to_u32, 3)
    .dispatch_1d(HyNomasBF16PatchProbe::ROWS * HyNomasBF16PatchProbe::CHANNELS)
  linear_plan.encode(chain)
  allocations = ML::MetalBuffer.hy_text_probe_allocations - before_allocations
  reads = ML::MetalBuffer.hy_text_probe_readbacks - before_reads
  unless allocations == 0 && reads == 0 && chain.wait_count == before_waits && before_waits == 0
    raise "BF16 patch encode allocated, read back, or waited"
  end
  chain.commit_and_wait
  raise "BF16 patch diagnostic requires one terminal wait" unless chain.wait_count == 1

  rounded_input_bytes = Bytes.new(input.contents.as(UInt8*), input.size.to_i32)
  bf16_bytes = Bytes.new(output_bf16.contents.as(UInt8*), output_bf16.size.to_i32)
  f32_bytes = Bytes.new(output_f32.contents.as(UInt8*), output_f32.size.to_i32)
  HyNomasGeneration.write_fresh_output(payload) do |io|
    io.write(rounded_input_bytes)
    io.write(bf16_bytes)
    io.write(f32_bytes)
  end
  records = {
    "patch.input_bf16_f32" => {dtype: "F32", shape: [1_i64, HyNomasBF16PatchProbe::CHANNELS.to_i64,
                                                     HyNomasBF16PatchProbe::FRAMES.to_i64, HyNomasBF16PatchProbe::HEIGHT.to_i64,
                                                     HyNomasBF16PatchProbe::WIDTH.to_i64], offset: 0_i64, nbytes: input.size,
                               sha256: Digest::SHA256.hexdigest(rounded_input_bytes)},
    "patch.output_bf16" => {dtype: "BF16", shape: [1_i64, HyNomasBF16PatchProbe::ROWS.to_i64, HyNomasBF16PatchProbe::DIM.to_i64],
                            offset: input.size, nbytes: output_bf16.size, sha256: Digest::SHA256.hexdigest(bf16_bytes)},
    "patch.output_f32" => {dtype: "F32", shape: [1_i64, HyNomasBF16PatchProbe::ROWS.to_i64, HyNomasBF16PatchProbe::DIM.to_i64],
                           offset: input.size + output_bf16.size, nbytes: output_f32.size,
                           sha256: Digest::SHA256.hexdigest(f32_bytes)},
  }
  payload_sha = HyNomasBF16PatchProbe.sha256(payload)
  HyNomasGeneration.write_fresh_output(output) do |io|
    JSON.build(io, indent: "  ") do |json|
      json.object do
        json.field "schema", HyNomasBF16PatchProbe::OUTPUT_SCHEMA
        json.field "case_manifest", File.realpath(manifest)
        json.field "case_manifest_sha256", manifest_sha
        json.field "case_payload_sha256", case_doc["payload_sha256"].as_s
        json.field "capture_manifest_sha256", case_doc["capture_manifest_sha256"].as_s
        json.field "weight_index_sha256", index.sha256
        json.field "global_weights_sha256", index.globals_bf16_sha256
        json.field "upstream_revision", HyNomasBF16PatchProbe::PINNED_REVISION
        json.field "geometry", {batches: 1, frames: HyNomasBF16PatchProbe::FRAMES,
                                height: HyNomasBF16PatchProbe::HEIGHT, width: HyNomasBF16PatchProbe::WIDTH,
                                rows: HyNomasBF16PatchProbe::ROWS, input_width: HyNomasBF16PatchProbe::CHANNELS,
                                output_width: HyNomasBF16PatchProbe::DIM}
        json.field "operation", HyNomasBF16PatchProbe::OPERATION
        json.field "input_rounding", "explicit Metal BF16 RNE in-place on F32 storage before production channel-to-token transpose"
        json.field "input_nonzero_low16_count", case_doc["input_nonzero_low16_count"].as_i64
        json.field "input_rounding_changed_count", case_doc["input_rounding_changed_count"].as_i64
        json.field "variant", variant.label
        json.field "named_bytes", named_bytes
        json.field "named_allocations_during_encode", allocations
        json.field "activation_readbacks_during_encode", reads
        json.field "terminal_wait_count", chain.wait_count
        json.field "gpu_elapsed_seconds", chain.gpu_elapsed_seconds
        json.field "payload", File.basename(payload)
        json.field "payload_bytes", input.size + output_bf16.size + output_f32.size
        json.field "payload_sha256", payload_sha
        json.field "tensors", records
      end
    end
  end
  puts "BF16 patch capture: #{output} (#{HyNomasBF16PatchProbe.sha256(output)})"
ensure
  Qwen25PrefixProbe.cleanup_after_drain(chain) do
    plan.try { |active| active.release(chain) }
    prepared.try(&.release)
    output_arena.release
    weight_arena.release
    input_arena.release
    files.reverse_each(&.close)
  end
end
