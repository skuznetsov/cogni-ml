require "option_parser"
require "json"
require "digest/sha256"
require "./support/hy_nomas_generation"
require "./support/qwen25_prefix_probe"
require "./support/hy_nomas_probe_instrumentation"
require "../src/ml/video/hy_nomas_mps_bf16_linear"

module HyNomasBF16LinearProbe
  extend self

  SCHEMA                = "cogni_ml.hy_nomas_bf16_linear_case/v1"
  OUTPUT_SCHEMA         = "cogni_ml.hy_nomas_bf16_linear_capture/v1"
  PINNED_REVISION       = "911cea6458a89c2663513ff7099561ec190dd491"
  INPUT_ELEMENTS        =   2048_i64
  OUTPUT_ELEMENTS       = 12_288_i64
  INPUT_BYTES           = INPUT_ELEMENTS * sizeof(Float32)
  OUTPUT_BF16_BYTES     = OUTPUT_ELEMENTS * 2
  OUTPUT_BYTES          = OUTPUT_ELEMENTS * sizeof(Float32)
  BF16_MODULATION_BYTES = (INPUT_ELEMENTS * OUTPUT_ELEMENTS + OUTPUT_ELEMENTS) * 2
  MAX_NAMED_BYTES       = 128_i64 * 1024**2

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

  def read_record!(case_doc : JSON::Any, payload_path : String,
                   name : String, shape : Array(Int64), offset : Int64,
                   nbytes : Int64, payload_bytes : Int64) : Bytes
    record = case_doc["tensors"][name]
    unless record["dtype"].as_s == "F32" &&
           record["shape"].as_a.map(&.as_i64) == shape &&
           record["offset"].as_i64 == offset &&
           record["nbytes"].as_i64 == nbytes &&
           offset >= 0 && nbytes > 0 && offset <= payload_bytes &&
           nbytes <= payload_bytes - offset && valid_digest?(record["sha256"].as_s)
      raise ArgumentError.new("BF16 linear case tensor descriptor mismatch: #{name}")
    end
    bytes = Bytes.new(nbytes.to_i32)
    File.open(payload_path) do |io|
      io.seek(offset)
      io.read_fully(bytes)
    end
    unless Digest::SHA256.hexdigest(bytes) == record["sha256"].as_s
      raise ArgumentError.new("BF16 linear case tensor digest mismatch: #{name}")
    end
    bytes
  end

  def parse_case!(path : String, supplied_sha : String) : Tuple(JSON::Any, String, Bytes)
    canonical_file(path, "linear case manifest")
    unless valid_digest?(supplied_sha) && sha256(path) == supplied_sha
      raise ArgumentError.new("BF16 linear case manifest digest mismatch")
    end
    doc = JSON.parse(File.read(path))
    unless doc["schema"].as_s == SCHEMA &&
           doc["upstream_revision"].as_s == PINNED_REVISION &&
           doc["geometry"]["input_width"].as_i == 2048 &&
           doc["geometry"]["output_width"].as_i == 12_288 &&
           doc["operation"].as_s == "pinned img_mod.act(vec.to_bfloat16); img_mod.linear(silu)" &&
           doc["torch_device"].as_s == "mps" &&
           doc["torch_input_dtype"].as_s == "bfloat16" &&
           doc["weight_index_sha256"].as_s == HyNomasGeneration::INDEX_SHA &&
           valid_digest?(doc["block_weights_sha256"].as_s)
      raise ArgumentError.new("BF16 linear case source or geometry contract mismatch")
    end
    payload_name = doc["payload"].as_s
    unless Path[payload_name].basename == payload_name && payload_name.ends_with?(".tensors.bin")
      raise ArgumentError.new("BF16 linear case payload must be a sibling tensor file")
    end
    payload_path = File.join(File.dirname(path), payload_name)
    canonical_file(payload_path, "linear case payload")
    payload_bytes = doc["payload_bytes"].as_i64
    payload_sha = doc["payload_sha256"].as_s
    unless payload_bytes == INPUT_BYTES + INPUT_BYTES + OUTPUT_BYTES &&
           valid_digest?(payload_sha) && File.size(payload_path) == payload_bytes &&
           sha256(payload_path) == payload_sha
      raise ArgumentError.new("BF16 linear case payload extent or digest mismatch")
    end
    offset = 0_i64
    read_record!(doc, payload_path, "input.vec", [1_i64, INPUT_ELEMENTS], offset,
      INPUT_BYTES, payload_bytes)
    offset += INPUT_BYTES
    input = read_record!(doc, payload_path, "linear.input", [1_i64, INPUT_ELEMENTS], offset,
      INPUT_BYTES, payload_bytes)
    offset += INPUT_BYTES
    read_record!(doc, payload_path, "linear.expected", [1_i64, OUTPUT_ELEMENTS], offset,
      OUTPUT_BYTES, payload_bytes)
    {doc, payload_path, input}
  end
end

manifest = ""
manifest_sha = ""
output = ""
variant_name = ""
budget = 0_i64
OptionParser.parse do |parser|
  parser.banner = "One actual BF16 img_mod.linear operation on the authenticated first-step vector."
  parser.on("--case PATH", "Torch-MPS BF16 modulation linear case manifest") { |v| manifest = v }
  parser.on("--case-sha256 SHA", "Trusted case manifest SHA-256") { |v| manifest_sha = v }
  parser.on("--output PATH", "New native capture JSON and sibling tensor sidecar") { |v| output = v }
  parser.on("--variant NAME", "bf16-graph or f32-accum") { |v| variant_name = v }
  parser.on("--max-named-bytes N", "Aggregate named Metal buffer budget (at most 128 MiB)") { |v| budget = v.to_i64 }
end

unless ARGV.empty? && !manifest.empty? && !manifest_sha.empty? && !output.empty? &&
       output.ends_with?(".json") && !variant_name.empty? &&
       0 < budget <= HyNomasBF16LinearProbe::MAX_NAMED_BYTES
  raise ArgumentError.new("case, case digest, fresh .json output, explicit graph variant and a budget up to 128 MiB are required")
end
variant = ML::Video::HyNomasMPSBF16Linear::Variant.parse(variant_name)
required_named_bytes = HyNomasBF16LinearProbe::BF16_MODULATION_BYTES +
                       HyNomasBF16LinearProbe::INPUT_BYTES +
                       HyNomasBF16LinearProbe::OUTPUT_BF16_BYTES +
                       HyNomasBF16LinearProbe::OUTPUT_BYTES
raise ArgumentError.new("BF16 linear named buffers exceed the supplied budget") if required_named_bytes > budget
payload = output.sub(/\.json\z/, ".tensors.bin")
{output, payload}.each do |path|
  raise ArgumentError.new("refusing to replace output #{path}") if File.exists?(path) || File.symlink?(path)
end

case_doc, case_payload_path, case_input = HyNomasBF16LinearProbe.parse_case!(manifest, manifest_sha)
index = HyNomasGeneration.authenticate_bf16_index
pin = index.blocks.first? || raise ArgumentError.new("authenticated BF16 cache has no block 0")
unless pin.block_index == 0 && pin.bf16_sha256 == case_doc["block_weights_sha256"].as_s &&
       index.sha256 == case_doc["weight_index_sha256"].as_s
  raise ArgumentError.new("BF16 linear case and native block-0 weight identity differ")
end

config = HyNomasGeneration::CONFIG
bindings = ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(
  config, ML::GGUF::SafeTensorDType::BF16, 0).select do |binding|
  binding.target == "img_mod_weight" || binding.target == "img_mod_bias"
end
unless bindings.size == 2 && bindings.map(&.target).sort == ["img_mod_bias", "img_mod_weight"]
  raise ArgumentError.new("could not select exactly the block-0 image modulation weight and bias")
end

input_arena = ML::Video::Arena.new
weight_arena = ML::Video::Arena.new
output_arena = ML::Video::Arena.new
files = [] of ML::Video::CheckpointFile
prepared : ML::Video::PreparedWeights? = nil
plan : ML::Video::HyNomasMPSBF16Linear::Plan? = nil
chain : ML::Metal::MPSCommandChain? = nil
begin
  block_file = ML::Video::BF16WeightCache.authenticate(pin.bf16_path, pin.bf16_sha256)
  files << block_file
  weight_plan = ML::Video::WeightPlan.new([block_file], bindings, budget,
    resident_dtype: ML::Video::WeightDType::BF16, require_all: false)
  unless weight_plan.entries.size == 2 && weight_plan.destination_bytes == HyNomasBF16LinearProbe::BF16_MODULATION_BYTES &&
         weight_plan.staging_bytes == 0 && !weight_plan.allow_f32_to_bf16
    raise ArgumentError.new("BF16 modulation weight plan is not an exact direct-copy pair")
  end
  weights_prepared = ML::Video::PreparedWeights.new(weight_plan, weight_arena)
  prepared = weights_prepared
  linear_weights = ML::Video::HyNomasDualStreamBlock::LinearWeights.new(weight_arena, "img_mod")
  input = input_arena.alloc("linear.input", HyNomasBF16LinearProbe::INPUT_ELEMENTS.to_i32)
  input.write_bytes(case_input.to_unsafe, case_input.size)
  output_bf16_buffer = output_arena.alloc_weight(
    "linear.output_bf16", HyNomasBF16LinearProbe::OUTPUT_ELEMENTS.to_i32,
    ML::Video::WeightDType::BF16)
  output_buffer = output_arena.alloc("linear.output", HyNomasBF16LinearProbe::OUTPUT_ELEMENTS.to_i32)
  named_bytes = input_arena.resident_bytes + weight_arena.resident_bytes + output_arena.resident_bytes
  unless named_bytes == HyNomasBF16LinearProbe::BF16_MODULATION_BYTES +
                        HyNomasBF16LinearProbe::INPUT_BYTES +
                        HyNomasBF16LinearProbe::OUTPUT_BF16_BYTES +
                        HyNomasBF16LinearProbe::OUTPUT_BYTES &&
         named_bytes <= budget
    raise ArgumentError.new("BF16 linear named resident bytes exceed the supplied budget")
  end
  plan = ML::Video::HyNomasMPSBF16Linear::Plan.new(
    input, linear_weights, output_bf16_buffer, output_buffer, variant)
  chain = ML::Metal::MPSCommandChain.new
  weights_prepared.encode(chain.encoder)
  before_allocations = ML::MetalBuffer.hy_text_probe_allocations
  before_reads = ML::MetalBuffer.hy_text_probe_readbacks
  plan.encode(chain)
  allocations = ML::MetalBuffer.hy_text_probe_allocations - before_allocations
  reads = ML::MetalBuffer.hy_text_probe_readbacks - before_reads
  unless allocations == 0 && reads == 0 && chain.wait_count == 0
    raise "BF16 linear encode allocated, read back, or waited"
  end
  chain.commit_and_wait
  raise "BF16 linear diagnostic requires one terminal wait" unless chain.wait_count == 1

  records = {} of String => NamedTuple(dtype: String, shape: Array(Int64), offset: Int64, nbytes: Int64, sha256: String)
  bytes = Bytes.new(output_buffer.contents.as(UInt8*), output_buffer.size.to_i32)
  HyNomasGeneration.write_fresh_output(payload) { |io| io.write(bytes) }
  records["linear.output"] = {dtype: "F32", shape: [1_i64, HyNomasBF16LinearProbe::OUTPUT_ELEMENTS],
                              offset: 0_i64, nbytes: output_buffer.size, sha256: Digest::SHA256.hexdigest(bytes)}
  payload_sha = HyNomasBF16LinearProbe.sha256(payload)
  HyNomasGeneration.write_fresh_output(output) do |io|
    JSON.build(io, indent: "  ") do |json|
      json.object do
        json.field "schema", HyNomasBF16LinearProbe::OUTPUT_SCHEMA
        json.field "case_manifest", File.realpath(manifest)
        json.field "case_manifest_sha256", manifest_sha
        json.field "case_payload_sha256", case_doc["payload_sha256"].as_s
        json.field "weight_index_sha256", index.sha256
        json.field "block_weights_sha256", pin.bf16_sha256
        json.field "upstream_revision", HyNomasBF16LinearProbe::PINNED_REVISION
        json.field "geometry", {rows: 1, input_width: 2048, output_width: 12_288}
        json.field "operation", "resident BF16 img_mod.linear weights+bias; F32 storage input cast to BF16"
        json.field "variant", variant.label
        json.field "named_bytes", named_bytes
        json.field "named_allocations_during_encode", allocations
        json.field "activation_readbacks_during_encode", reads
        json.field "terminal_wait_count", chain.wait_count
        json.field "gpu_elapsed_seconds", chain.gpu_elapsed_seconds
        json.field "payload", File.basename(payload)
        json.field "payload_bytes", output_buffer.size
        json.field "payload_sha256", payload_sha
        json.field "tensors", records
      end
    end
  end
  puts "BF16 linear capture: #{output} (#{HyNomasBF16LinearProbe.sha256(output)})"
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
