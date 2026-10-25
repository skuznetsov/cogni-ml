require "option_parser"
require "./support/qwen25_prefix_probe"
require "../src/ml/gguf/qwen25_vl_embedding"

# Probe-only cumulative instrumentation also catches allocations that were
# released before the end of encoding. It does not count opaque MPS resources.
class ML::MetalBuffer
  @@qwen25_probe_allocations = 0_i64

  def self.qwen25_probe_allocations : Int64
    @@qwen25_probe_allocations
  end

  private def track_alloc(bytes : Int64) : Nil
    @@qwen25_probe_allocations += 1
    previous_def(bytes)
  end
end

checkpoint_root = ""
reference = ""
reference_sha256 = ""
report_path = ""
max_named_bytes = 0_i64
rms_power_fed = false
OptionParser.parse do |parser|
  parser.banner = "Bounded pretrained Qwen2.5-VL prefix qualification; use the external memory guard."
  parser.on("--checkpoint-root PATH", "Pinned local safetensors checkpoint") { |v| checkpoint_root = v }
  parser.on("--reference PATH", "Authenticated upstream prefix manifest") { |v| reference = v }
  parser.on("--reference-sha256 SHA256", "Trusted digest from a successful upstream oracle run") { |v| reference_sha256 = v }
  parser.on("--report PATH", "New native comparison report") { |v| report_path = v }
  parser.on("--max-named-bytes N", "Aggregate named-buffer admission cap") { |v| max_named_bytes = v.to_i64 }
  parser.on("--rms-power-fed", "Use a prepared GPU exponent for both RMS sites in every prefix block") { rms_power_fed = true }
end
raise ArgumentError.new("checkpoint, reference, trusted digest, report and positive named-buffer budget required") if checkpoint_root.empty? || reference.empty? || reference_sha256.empty? || report_path.empty? || max_named_bytes <= 0
raise ArgumentError.new("refusing to replace native report") if File.exists?(report_path) || File.symlink?(report_path)
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?

fixture = Qwen25PrefixProbe::Fixture.new(reference, checkpoint_root, reference_sha256)
config = fixture.config
tokens = fixture.tokens.size
unique_ids = fixture.tokens.uniq.sort
gather_sizes = ML::GGUF::Qwen25VLEmbeddingPlan.resident_bytes(config, tokens, unique_ids.size)
prefix_bytes = ML::GGUF::Qwen25VLPrefixPlan.named_bytes(config, tokens,
  Qwen25PrefixProbe::PREFIX_LAYERS, rms_power_fed: rms_power_fed)
named_bytes = prefix_bytes + gather_sizes.compact_rows + gather_sizes.remapped_ids
raise ArgumentError.new("pretrained prefix exceeds aggregate named-buffer budget") unless named_bytes <= max_named_bytes
STDERR.puts "admitted #{named_bytes} named bytes, #{tokens} tokens, 26 BF16 decoder blocks"
weights = ML::GGUF::Qwen25VLWeights.from_directory(checkpoint_root)
owned = [] of ML::MetalBuffer
layer_weights = [] of Hash(String, ML::MetalBuffer)
embedding : ML::GGUF::Qwen25VLEmbeddingPlan? = nil
prefix : ML::GGUF::Qwen25VLPrefixPlan? = nil
chain : ML::Metal::MPSCommandChain? = nil
begin
  raw_rows = fixture.verify_embedding_rows(weights)
  table = ML::MetalBuffer.new(raw_rows.size.to_i64)
  owned << table
  table.write_bytes(raw_rows.to_unsafe, raw_rows.size)
  indices = fixture.tokens.map { |id| unique_ids.index(id).not_nil! }
  embedding = ML::GGUF::Qwen25VLEmbeddingPlan.new(config, tokens,
    compact_rows: table, row_indices: indices, max_named_bytes: gather_sizes.total)
  mask_words = Array(UInt16).new(tokens * tokens) do |i|
    query, key = i // tokens, i % tokens
    key <= query && fixture.attention_mask[key] == 1 ? 0_u16 : 0xff7f_u16
  end
  mask = ML::MetalBuffer.new(mask_words.size.to_i64 * 2)
  owned << mask
  mask.write_bytes(mask_words.to_unsafe.as(UInt8*), mask_words.size * 2)
  positions = ML::MetalBuffer.new(tokens.to_i64 * 4)
  owned << positions
  positions.write(fixture.positions)
  inv_freq = ML::MetalBuffer.new(fixture.inv_freq.size.to_i64 * 4)
  owned << inv_freq
  inv_freq.write(fixture.inv_freq)
  26.times do |index|
    set = weights.upload_block_weights(index, fixture.tensor_sha256s)
    layer_weights << set
    set.each_value { |buffer| owned << buffer }
    STDERR.puts "loaded native decoder weights #{index + 1}/26" if (index + 1) % 4 == 0 || index == 25
  end
  prefix = ML::GGUF::Qwen25VLPrefixPlan.new(config, tokens, embedding.output, mask,
    positions, inv_freq, layer_weights, prefix_bytes, rms_power_fed: rms_power_fed)
  raise "prefix RMS mode was not applied to every block" unless prefix.blocks.all? { |block| block.rms_power_fed == rms_power_fed }
  rms_fed_blocks = prefix.blocks.count(&.rms_power_fed)
  chain = ML::Metal::MPSCommandChain.new
  allocations_before = ML::MetalBuffer.qwen25_probe_allocations
  encode_started = Time.instant
  embedding.encode(chain)
  prefix.encode(chain)
  encode_seconds = (Time.instant - encode_started).total_seconds
  encode_allocations = ML::MetalBuffer.qwen25_probe_allocations - allocations_before
  raise "encoding allocated named Metal buffers" unless encode_allocations == 0
  raise "encoding introduced a device wait" unless chain.wait_count == 0
  early_release_rejected = false
  begin
    prefix.release
  rescue ex : ArgumentError
    raise ex unless ex.message.try(&.includes?("terminal wait required"))
    early_release_rejected = true
  end
  raise "in-flight prefix release was accepted" unless early_release_rejected
  wait_started = Time.instant
  chain.commit_and_wait
  wait_seconds = (Time.instant - wait_started).total_seconds
  raise "expected exactly one terminal device wait" unless chain.wait_count == 1
  comparisons = [fixture.compare(embedding.output, 0)]
  prefix.blocks.each_with_index { |block, index| comparisons << fixture.compare(block.output, index + 1) }
  failures = comparisons.select { |row| row[:failures] > 0 }
  File.write(report_path, JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", "cogni_ml.qwen25_vl_pretrained_prefix_native_comparison"
      json.field "schema_version", 1
      json.field "reference", fixture.manifest_path
      json.field "reference_sha256", fixture.manifest_sha256
      json.field "checkpoint_revision", ML::GGUF::Qwen25VLWeights::EXPECTED_MODEL_REVISION
      json.field "tokens", tokens
      json.field "prefix_layers", 26
      json.field "rms_power_fed", rms_power_fed
      json.field "rms_fed_block_count", rms_fed_blocks
      json.field "rms_statistics_sites_using_fed_exponent", 2 * rms_fed_blocks
      json.field "rms_exponent_named_bytes", Qwen25PrefixProbe::PREFIX_LAYERS.to_i64 * ML::GGUF::Qwen25VLBlockPlan.rms_exponent_bytes(rms_power_fed)
      json.field "named_bytes", named_bytes
      json.field "shared_workspace_bytes", ML::GGUF::Qwen25VLPrefixPlan.workspace_bytes(config, tokens, 26)
      json.field "shared_stage_scratch_bytes", prefix.workspace.bytes
      json.field "retained_hidden_output_bytes", Qwen25PrefixProbe::PREFIX_LAYERS.to_i64 * tokens * config.dim * 2
      json.field "gather_bytes", {compact_rows: gather_sizes.compact_rows,
                                  remapped_ids: gather_sizes.remapped_ids,
                                  output: gather_sizes.output, total: gather_sizes.total}
      json.field "named_allocations_during_encode", encode_allocations
      json.field "terminal_wait_count", chain.wait_count
      json.field "early_release_rejected", early_release_rejected
      json.field "diagnostic_readbacks_after_terminal_wait", comparisons.size
      json.field "activation_readbacks_before_terminal_wait", 0
      json.field "encode_seconds", encode_seconds
      json.field "terminal_wait_seconds", wait_seconds
      json.field "gpu_elapsed_seconds", chain.gpu_elapsed_seconds
      json.field "command_segments", chain.segment_count
      json.field "tolerance", {atol: Qwen25PrefixProbe::ATOL, rtol: Qwen25PrefixProbe::RTOL}
      json.field "comparisons", comparisons
      json.field "passed", failures.empty?
      json.field "first_mismatch_state", failures.first?.try(&.[:state])
    end
  end)
  comparisons.each { |row| puts row.to_json }
  raise "pretrained prefix first mismatch at captured state #{failures.first[:state]}" unless failures.empty?
  puts "PASS: embedding + all 26 pretrained BF16 hidden states; one terminal wait; zero named allocations during encode"
ensure
  # Even partial encode failures must drain before any borrowed source or
  # workspace can be released. A completed terminal error still authorizes
  # cleanup, while an incomplete drain deliberately retains resident buffers.
  Qwen25PrefixProbe.cleanup_after_drain(chain) do
    prefix.try(&.release)
    embedding.try(&.release)
    chain.try(&.release)
    owned.reverse_each { |buffer| buffer.release if buffer.valid? }
    weights.close
  end
end
