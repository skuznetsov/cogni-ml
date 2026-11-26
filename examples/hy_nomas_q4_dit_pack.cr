require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/hy_nomas_q4_dit_io"

# Offline derivative creation: one block of F32 source staging at a time.
# Inference never reads these raw source tensors or persists GPU activations.
source_path, directory = "", ""
budget = 1536_i64 * 1024 * 1024
OptionParser.parse do |parser|
  parser.on("--source PATH", "Authenticated complete pinned DiT checkpoint") { |value| source_path = value }
  parser.on("--output-dir PATH", "New empty cache directory on the external SSD") { |value| directory = value }
  parser.on("--budget-mib N", "Bounded per-block named setup budget including pack outputs") { |value| budget = value.to_i64 * 1024 * 1024 }
end
unless !source_path.empty? && !directory.empty? && 0 < budget <= 2048_i64 * 1024 * 1024
  raise ArgumentError.new("Expected source/output paths and budget in 1..2048 MiB")
end
artifact_root = File.realpath("/opt/mldata_ssd/cogni-ml/hy-nomas-q4-dit-20261025")
output_parent = File.realpath(File.dirname(directory))
unless (output_parent == artifact_root || output_parent.starts_with?(artifact_root + "/")) &&
       !File.info?(directory, false)
  raise ArgumentError.new("Packer requires a new directory under the admitted external SSD artifact root")
end
source = HyNomasQ4DiTIO.authenticate_source!(source_path)
Dir.mkdir(directory, 0o700)
blocks = [] of JSON::Any
globals = nil.as(JSON::Any?)
receipts = [] of JSON::Any
begin
  (-1...HyNomasQ4DiTIO::BLOCKS).each do |index|
    bindings = index < 0 ? HyNomasQ4DiTIO::GLOBAL_BINDINGS : ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(HyNomasQ4DiTIO::CONFIG, block_index: index)
    q4_reserve_bytes = if index < 0
                         0_i64
                       else
                         ML::Video::HyNomasQ4Cache::PackBlock.resident_bytes_for(
                           HyNomasQ4DiTIO::CONFIG, index)
                       end
    reserve_bytes = q4_reserve_bytes + HyNomasQ4DiTIO::BF16_FINITE_STATUS_BYTES
    plan = ML::Video::WeightPlan.new([source], bindings, budget,
      reserve_bytes: reserve_bytes,
      resident_dtype: ML::Video::WeightDType::BF16,
      allow_f32_to_bf16: true, require_all: false)
    arena = ML::Video::Arena.new
    prepared = nil.as(ML::Video::PreparedWeights?)
    finite_check = nil.as(HyNomasQ4DiTIO::BF16FiniteCheck?)
    packer = nil.as(ML::Video::HyNomasQ4Cache::PackBlock?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      prepared = ML::Video::PreparedWeights.new(plan, arena)
      finite_check = HyNomasQ4DiTIO::BF16FiniteCheck.new(plan, arena)
      if index >= 0
        packer = ML::Video::HyNomasQ4Cache::PackBlock.new(HyNomasQ4DiTIO::CONFIG, index,
          HyNomasQ4DiTIO::SOURCE_SHA, HyNomasQ4DiTIO::SOURCE_BYTES, plan, arena)
      end
      chain = ML::Metal::MPSCommandChain.new
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      reads = ML::MetalBuffer.hy_text_probe_readbacks
      prepared.encode(chain.encoder)
      finite_check.encode(chain)
      packer.try(&.encode_pack(chain))
      unless ML::MetalBuffer.hy_text_probe_allocations == allocations &&
             ML::MetalBuffer.hy_text_probe_readbacks == reads && chain.wait_count == 0
        raise "Offline GPU conversion/pack allocated or synchronized during encode"
      end
      chain.commit_and_wait
      finite_check.validate_after_terminal!(chain)
      bf16_name = index < 0 ? "globals-bf16.safetensors" : "block-#{index}-bf16.safetensors"
      bf16_sha = HyNomasQ4DiTIO.write_bf16_selection!(File.join(directory, bf16_name), plan, arena, chain)
      if index < 0
        globals = JSON.parse({bf16: bf16_name, sha256: bf16_sha}.to_json)
      else
        published = packer.not_nil!.write_after_terminal!(File.join(directory, "block-#{index}-q4k.json"), chain)
        blocks << JSON.parse({block_index: index, manifest: File.basename(published.manifest_path),
                              manifest_sha256: published.manifest_sha256, payload_sha256: published.payload_sha256,
                              bf16: bf16_name, bf16_sha256: bf16_sha}.to_json)
      end
      receipts << JSON.parse({block_index: index, destination_bytes: plan.destination_bytes,
                              f32_source_staging_bytes: plan.staging_bytes,
                              reserved_q4_pack_bytes: q4_reserve_bytes,
                              bf16_finite_status_bytes: HyNomasQ4DiTIO::BF16_FINITE_STATUS_BYTES,
                              total_reserved_setup_bytes: reserve_bytes,
                              admitted_peak_bytes: plan.peak_bytes,
                              encode_named_allocations: ML::MetalBuffer.hy_text_probe_allocations - allocations,
                              preterminal_readbacks: 0, terminal_waits: chain.wait_count}.to_json)
      STDERR.puts "packed #{index < 0 ? "globals" : "block #{index}/53"}"
    ensure
      chain.try(&.commit_and_wait)
      packer.try(&.release)
      finite_check.try(&.release)
      prepared.try(&.release)
      chain.try(&.release)
      arena.release
    end
  end
  index_doc = {schema: HyNomasQ4DiTIO::INDEX_SCHEMA, schema_version: 1,
               source: {sha256: HyNomasQ4DiTIO::SOURCE_SHA, bytes: HyNomasQ4DiTIO::SOURCE_BYTES},
               blocks: blocks, globals: globals.not_nil!}
  index_path = File.join(directory, "index.json")
  HyNomasQ4DiTIO.write_new!(index_path, index_doc.to_pretty_json + "\n")
  HyNomasQ4DiTIO.write_new!(File.join(directory, "pack-receipt.json"),
    {schema: "hy-nomas-q4-dit-pack-receipt-v1", index_sha256: HyNomasQ4DiTIO.sha256_file(index_path),
     packing: "GPU F32 to BF16 then Q4_K eight local ranges; vectors BF16; globals BF16", phases: receipts}.to_pretty_json + "\n")
  puts({index: index_path, index_sha256: HyNomasQ4DiTIO.sha256_file(index_path), blocks: blocks.size}.to_json)
ensure
  source.close
end
