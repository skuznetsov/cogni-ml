require "option_parser"
require "./support/qwen25_prefix_probe"
require "../src/ml/gguf/qwen25_vl_embedding"
require "../src/ml/gguf/qwen25_hy_nomas_prompt"
require "../src/ml/gguf/qwen25_hy_nomas_conditioning"
require "./support/hy_nomas_text_session"

# Probe-only cumulative instrumentation also catches allocations that were
# released before the end of encoding. It does not count opaque MPS resources.
class ML::MetalBuffer
  @@qwen25_probe_allocations = 0_i64
  @@qwen25_probe_readbacks = 0_i64

  def self.qwen25_probe_allocations : Int64
    @@qwen25_probe_allocations
  end

  def self.qwen25_probe_readbacks : Int64
    @@qwen25_probe_readbacks
  end

  def read_bytes(ptr : Pointer(UInt8), byte_size : Int) : Nil
    @@qwen25_probe_readbacks += 1
    previous_def(ptr, byte_size)
  end

  def read(count : Int32) : Array(Float32)
    @@qwen25_probe_readbacks += 1
    previous_def(count)
  end

  def read_to_slice(dest : Slice(Float32)) : Nil
    @@qwen25_probe_readbacks += 1
    previous_def(dest)
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
prompt_file = ""
hy_reference = ""
hy_reference_sha256 = ""
OptionParser.parse do |parser|
  parser.banner = "Bounded pretrained Qwen2.5-VL prefix qualification; use the external memory guard."
  parser.on("--checkpoint-root PATH", "Pinned local safetensors checkpoint") { |v| checkpoint_root = v }
  parser.on("--reference PATH", "Authenticated upstream prefix manifest") { |v| reference = v }
  parser.on("--reference-sha256 SHA256", "Trusted digest from a successful upstream oracle run") { |v| reference_sha256 = v }
  parser.on("--report PATH", "New native comparison report") { |v| report_path = v }
  parser.on("--max-named-bytes N", "Aggregate named-buffer admission cap") { |v| max_named_bytes = v.to_i64 }
  parser.on("--rms-power-fed", "Use a prepared GPU exponent for both RMS sites in every prefix block") { rms_power_fed = true }
  parser.on("--hy-nomas-prompt-file PATH", "Prepare a natural video prompt natively and append its GPU output crop") { |v| prompt_file = v }
  parser.on("--hy-nomas-reference PATH", "Append the qualified text refiner and one bounded DiT block") { |v| hy_reference = v }
  parser.on("--hy-nomas-reference-sha256 SHA256", "Trusted fresh refiner/block oracle digest") { |v| hy_reference_sha256 = v }
end
raise ArgumentError.new("checkpoint, reference, trusted digest, report and positive named-buffer budget required") if checkpoint_root.empty? || reference.empty? || reference_sha256.empty? || report_path.empty? || max_named_bytes <= 0
raise ArgumentError.new("refusing to replace native report") if File.exists?(report_path) || File.symlink?(report_path)
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?

fixture = Qwen25PrefixProbe::Fixture.new(reference, checkpoint_root, reference_sha256)
raise ArgumentError.new("refiner reference and trusted digest must be supplied together") unless hy_reference.empty? == hy_reference_sha256.empty?
raise ArgumentError.new("refiner handoff requires native prompt preparation") if !hy_reference.empty? && prompt_file.empty?
hy_fixture = hy_reference.empty? ? nil : HyNomasTextProbe::Fixture.new(hy_reference, hy_reference_sha256, fixture.manifest_sha256)
hy_fixture.try { |consumer| HyNomasTextProbe.validate_qwen_binding!(consumer.document, fixture.document) }
native_prompt = if metadata = fixture.prompt_metadata
                  raise ArgumentError.new("natural prompt fixture requires native --hy-nomas-prompt-file") if prompt_file.empty?
                  raise ArgumentError.new("prompt input exceeds 64 KiB") unless 0 < File.size(prompt_file) <= 64 * 1024
                  prompt_text = File.read(prompt_file)
                  raise ArgumentError.new("native prompt differs from authenticated text") unless prompt_text == metadata["prompt"].as_s
                  prepared = ML::GGUF::Qwen25HyNomasPrompt.new(checkpoint_root).prepare(prompt_text)
                  raise ArgumentError.new("native rendered prompt mismatch") unless prepared.rendered_prompt == metadata["rendered_prompt"].as_s
                  raise ArgumentError.new("native prompt token IDs mismatch") unless prepared.tokens == fixture.tokens
                  raise ArgumentError.new("native prompt mask mismatch") unless prepared.attention_mask == fixture.attention_mask
                  raise ArgumentError.new("native prompt positions mismatch") unless prepared.positions == fixture.positions
                  raise ArgumentError.new("native prompt crop mismatch") unless prepared.crop_start == fixture.crop_start && prepared.cropped_attention_mask == metadata["cropped_attention_mask"].as_a.map(&.as_i)
                  prepared
                else
                  raise ArgumentError.new("prompt input requires an authenticated prompt fixture") unless prompt_file.empty?
                  nil
                end
# The native-prepared arrays are used for uploads, after exact independent
# comparison with the upstream request. Natural mode is not fixture-IDs-in.
input_tokens = native_prompt ? native_prompt.tokens : fixture.tokens
input_mask = native_prompt ? native_prompt.attention_mask : fixture.attention_mask
input_positions = native_prompt ? native_prompt.positions : fixture.positions
config = fixture.config
tokens = input_tokens.size
unique_ids = input_tokens.uniq.sort
gather_sizes = ML::GGUF::Qwen25VLEmbeddingPlan.resident_bytes(config, tokens, unique_ids.size)
prefix_bytes = ML::GGUF::Qwen25VLPrefixPlan.named_bytes(config, tokens,
  Qwen25PrefixProbe::PREFIX_LAYERS, rms_power_fed: rms_power_fed)
conditioning_sizes = native_prompt ? ML::GGUF::Qwen25VLConditioningPlan.resident_bytes(config, tokens, native_prompt.crop_start) : nil
named_bytes = prefix_bytes + gather_sizes.compact_rows + gather_sizes.remapped_ids + (conditioning_sizes.try(&.total) || 0_i64)
hy_named_bytes = hy_fixture ? HyNomasTextProbe::Session.admit(hy_fixture, max_named_bytes) : 0_i64
named_bytes += hy_named_bytes
raise ArgumentError.new("pretrained prefix exceeds aggregate named-buffer budget") unless named_bytes <= max_named_bytes
STDERR.puts "admitted #{named_bytes} named bytes, #{tokens} tokens, 26 BF16 decoder blocks"
weights = ML::GGUF::Qwen25VLWeights.from_directory(checkpoint_root)
owned = [] of ML::MetalBuffer
layer_weights = [] of Hash(String, ML::MetalBuffer)
embedding : ML::GGUF::Qwen25VLEmbeddingPlan? = nil
prefix : ML::GGUF::Qwen25VLPrefixPlan? = nil
conditioning : ML::GGUF::Qwen25VLConditioningPlan? = nil
hy_session : HyNomasTextProbe::Session? = nil
chain : ML::Metal::MPSCommandChain? = nil
begin
  raw_rows = fixture.verify_embedding_rows(weights)
  table = ML::MetalBuffer.new(raw_rows.size.to_i64)
  owned << table
  table.write_bytes(raw_rows.to_unsafe, raw_rows.size)
  indices = input_tokens.map { |id| unique_ids.index(id).not_nil! }
  embedding = ML::GGUF::Qwen25VLEmbeddingPlan.new(config, tokens,
    compact_rows: table, row_indices: indices, max_named_bytes: gather_sizes.total)
  mask_words = Array(UInt16).new(tokens * tokens) do |i|
    query, key = i // tokens, i % tokens
    key <= query && input_mask[key] == 1 ? 0_u16 : 0xff7f_u16
  end
  mask = ML::MetalBuffer.new(mask_words.size.to_i64 * 2)
  owned << mask
  mask.write_bytes(mask_words.to_unsafe.as(UInt8*), mask_words.size * 2)
  positions = ML::MetalBuffer.new(tokens.to_i64 * 4)
  owned << positions
  positions.write(input_positions)
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
  if prepared = native_prompt
    crop_bytes = conditioning_sizes.not_nil!
    raw_mask = ML::MetalBuffer.new(crop_bytes.raw_mask)
    owned << raw_mask
    raw_mask.write_bytes(input_mask.to_unsafe.as(UInt8*), crop_bytes.raw_mask.to_i)
    conditioning = ML::GGUF::Qwen25VLConditioningPlan.new(config, tokens, prepared.crop_start,
      prefix.blocks.last.output, raw_mask, crop_bytes.total)
    if hy_data = hy_fixture
      hy_session = HyNomasTextProbe::Session.new(hy_data, conditioning.hidden,
        prepared.cropped_attention_mask, hy_named_bytes)
    end
  end
  chain = ML::Metal::MPSCommandChain.new
  allocations_before = ML::MetalBuffer.qwen25_probe_allocations
  reads_before = ML::MetalBuffer.qwen25_probe_readbacks
  encode_started = Time.instant
  embedding.encode(chain)
  prefix.encode(chain)
  # This is the SAME producer chain. No source-chain identity is inferred
  # from bare buffers by the crop plan.
  conditioning.try(&.encode(chain))
  hy_session.try(&.encode(chain))
  encode_seconds = (Time.instant - encode_started).total_seconds
  encode_allocations = ML::MetalBuffer.qwen25_probe_allocations - allocations_before
  raise "encoding allocated named Metal buffers" unless encode_allocations == 0
  reads_before_wait = ML::MetalBuffer.qwen25_probe_readbacks - reads_before
  raise "encoding read back activation bytes" unless reads_before_wait == 0
  raise "encoding introduced a device wait" unless chain.wait_count == 0
  early_release_rejected = false
  begin
    prefix.release
  rescue ex : ArgumentError
    raise ex unless ex.message.try(&.includes?("terminal wait required"))
    early_release_rejected = true
  end
  raise "in-flight prefix release was accepted" unless early_release_rejected
  conditioning_release_rejected = false
  if crop = conditioning
    begin
      crop.release
    rescue ex : ArgumentError
      raise ex unless ex.message.try(&.includes?("terminal wait required"))
      conditioning_release_rejected = true
    end
    raise "in-flight conditioning release was accepted" unless conditioning_release_rejected
  end
  hy_early_release_rejected = false
  if consumer = hy_session
    begin
      consumer.release
    rescue ex : ArgumentError
      raise ex unless ex.message.try(&.includes?("command chain"))
      hy_early_release_rejected = true
    end
    raise "in-flight refiner/block release was accepted" unless hy_early_release_rejected
  end
  wait_started = Time.instant
  chain.commit_and_wait
  wait_seconds = (Time.instant - wait_started).total_seconds
  raise "expected exactly one terminal device wait" unless chain.wait_count == 1
  comparisons = [fixture.compare(embedding.output, 0)]
  prefix.blocks.each_with_index { |block, index| comparisons << fixture.compare(block.output, index + 1) }
  conditioning_comparison = conditioning.try { |crop| fixture.compare_conditioning(crop.hidden, crop.mask) }
  hy_comparisons = hy_session.try(&.compare) || [] of HyNomasTextProbe::Comparison
  hy_passed = hy_comparisons.all? { |row| row[:failures] == 0 }
  crop_passed = conditioning_comparison.try { |row| row[:failures] == 0 && row[:mask_failures] == 0 } != false
  diagnostic_reads = ML::MetalBuffer.qwen25_probe_readbacks - reads_before
  raise "unexpected number of post-wait diagnostic reads" unless diagnostic_reads == comparisons.size + (conditioning ? 2 : 0) + hy_comparisons.size
  failures = comparisons.select { |row| row[:failures] > 0 }
  File.write(report_path, JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", "cogni_ml.qwen25_vl_pretrained_prefix_native_comparison"
      json.field "schema_version", 1
      json.field "reference", fixture.manifest_path
      json.field "reference_sha256", fixture.manifest_sha256
      json.field "checkpoint_revision", ML::GGUF::Qwen25VLWeights::EXPECTED_MODEL_REVISION
      json.field "tokens", tokens
      json.field "native_prompt_prepared", !native_prompt.nil?
      if prepared = native_prompt
        json.field "prompt" do
          json.object do
            json.field "prompt_sha256", fixture.prompt_metadata.not_nil!["prompt_sha256"].as_s
            json.field "rendered_prompt_sha256", Digest::SHA256.hexdigest(prepared.rendered_prompt)
            json.field "crop_start", prepared.crop_start
            json.field "conditioning_tokens", prepared.cropped_attention_mask.size
            json.field "visible_raw_tokens", input_mask.sum
            json.field "tokenizer_assets" do
              json.object do
                json.field "tokenizer.json", prepared.tokenizer_json_sha256
                json.field "tokenizer_config.json", prepared.tokenizer_config_sha256
              end
            end
          end
        end
      end
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
      if sizes = conditioning_sizes
        json.field "conditioning_bytes", {raw_mask: sizes.raw_mask, hidden: sizes.hidden, mask: sizes.mask, total: sizes.total}
      end
      json.field "named_allocations_during_encode", encode_allocations
      json.field "terminal_wait_count", chain.wait_count
      json.field "early_release_rejected", early_release_rejected
      json.field "conditioning_early_release_rejected", conditioning_release_rejected
      json.field "diagnostic_readbacks_after_terminal_wait", diagnostic_reads
      json.field "activation_readbacks_before_terminal_wait", reads_before_wait
      json.field "encode_seconds", encode_seconds
      json.field "terminal_wait_seconds", wait_seconds
      json.field "gpu_elapsed_seconds", chain.gpu_elapsed_seconds
      json.field "command_segments", chain.segment_count
      json.field "tolerance", {atol: Qwen25PrefixProbe::ATOL, rtol: Qwen25PrefixProbe::RTOL}
      json.field "comparisons", comparisons
      json.field "conditioning_comparison", conditioning_comparison
      if data = hy_fixture
        json.field "hy_nomas_handoff", {reference: data.path, reference_sha256: data.sha256,
                                        named_bytes: hy_named_bytes, early_release_rejected: hy_early_release_rejected,
                                        comparisons: hy_comparisons, passed: hy_passed,
                                        producer_and_consumer_share_chain: true,
                                        projected_text_buffer_is_block_input: true,
                                        precision_scope: "BF16 resident weights; F32 math and attention control"}
      end
      json.field "passed", failures.empty? && crop_passed && hy_passed
      json.field "first_mismatch_state", failures.first?.try(&.[:state])
    end
  end)
  comparisons.each { |row| puts row.to_json }
  raise "pretrained prefix first mismatch at captured state #{failures.first[:state]}" unless failures.empty?
  raise "pretrained conditioning crop mismatch" unless crop_passed
  raise "refiner/block first mismatch: #{hy_comparisons.find { |row| row[:failures] > 0 }.try(&.[:name])}" unless hy_passed
  puts "PASS: embedding + all 26 pretrained BF16 hidden states#{conditioning ? " + GPU conditioning crop" : ""}; one terminal wait; zero named allocations during encode"
ensure
  # Even partial encode failures must drain before any borrowed source or
  # workspace can be released. A completed terminal error still authorizes
  # cleanup, while an incomplete drain deliberately retains resident buffers.
  Qwen25PrefixProbe.cleanup_after_drain(chain) do
    hy_session.try(&.release)
    conditioning.try(&.release)
    prefix.try(&.release)
    embedding.try(&.release)
    chain.try(&.release)
    owned.reverse_each { |buffer| buffer.release if buffer.valid? }
    weights.close
  end
end
