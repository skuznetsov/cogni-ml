require "option_parser"
require "./support/qwen25_prefix_probe"
require "../src/ml/gguf/qwen25_vl_embedding"
require "../src/ml/gguf/qwen25_hy_nomas_prompt"
require "../src/ml/gguf/qwen25_hy_nomas_conditioning"
require "./support/hy_nomas_conditioned_four_probe"

checkpoint_root = ""
reference = ""
reference_sha = ""
conditioning_reference = ""
conditioning_sha = ""
prompt_file = ""
report = ""
caches = Array(String).new(4, "")
digests = Array(String).new(4, "")
budget = 0_i64
mode = "positive"
controls = 0
OptionParser.parse do |parser|
  parser.banner = "Live Qwen conditioning to four masked full-J Hy-Nomas blocks; external guard required."
  parser.on("--checkpoint-root PATH", "Pinned local Qwen checkpoint") { |v| checkpoint_root = v }
  parser.on("--reference PATH", "Authenticated upstream Qwen prefix reference") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted Qwen reference digest") { |v| reference_sha = v }
  parser.on("--conditioning-reference PATH", "Fresh refiner/four-block upstream reference") { |v| conditioning_reference = v }
  parser.on("--conditioning-reference-sha256 SHA", "Trusted refiner/four-block digest") { |v| conditioning_sha = v }
  parser.on("--hy-nomas-prompt-file PATH", "Authenticated natural video prompt") { |v| prompt_file = v }
  parser.on("--report PATH", "New native JSON report") { |v| report = v }
  parser.on("--max-named-bytes N", "Aggregate named buffer budget") { |v| budget = v.to_i64 }
  4.times do |i|
    parser.on("--cache#{i} PATH", "Qualified BF16 block #{i}") { |v| caches[i] = v }
    parser.on("--cache#{i}-sha256 SHA", "Trusted BF16 block #{i} digest") { |v| digests[i] = v }
  end
  parser.on("--padding-as-valid-control", "Require detection of all padding incorrectly admitted to DiT attention") { controls += 1; mode = "padding-as-valid" }
  parser.on("--block3-as-block2-control", "Require detection of block2 weights replacing block3") { controls += 1; mode = "block3-as-block2" }
end
unless ARGV.empty? && controls <= 1 && budget > 0 &&
       {checkpoint_root, reference, reference_sha, conditioning_reference, conditioning_sha, prompt_file, report}.all? { |v| !v.empty? } &&
       (caches + digests).all? { |v| !v.empty? }
  raise ArgumentError.new("ordered checkpoints/digests, prompt, fresh report, positive budget and at most one control required")
end
raise ArgumentError.new("refusing to replace native report") if File.exists?(report) || File.symlink?(report)

# All source authentication and named-byte admission precede Metal setup.
fixture = Qwen25PrefixProbe::Fixture.new(reference, checkpoint_root, reference_sha)
hy_fixture = HyNomasConditionedFourProbe::Fixture.new(conditioning_reference, conditioning_sha, fixture.manifest_sha256)
HyNomasTextProbe.validate_qwen_binding!(hy_fixture.document, fixture.document)
metadata = fixture.prompt_metadata || raise ArgumentError.new("natural prompt metadata required")
raise ArgumentError.new("prompt input must be a bounded regular file") unless File.file?(prompt_file) && !File.symlink?(prompt_file) && 0 < File.size(prompt_file) <= 64 * 1024
prompt_text = File.read(prompt_file)
raise ArgumentError.new("prompt differs from authenticated upstream request") unless prompt_text == metadata["prompt"].as_s
prepared = ML::GGUF::Qwen25HyNomasPrompt.new(checkpoint_root).prepare(prompt_text)
unless prepared.rendered_prompt == metadata["rendered_prompt"].as_s &&
       prepared.tokens == fixture.tokens && prepared.attention_mask == fixture.attention_mask &&
       prepared.positions == fixture.positions && prepared.crop_start == fixture.crop_start &&
       prepared.cropped_attention_mask == hy_fixture.mask
  raise ArgumentError.new("native prompt IDs/mask/positions/rendering/crop mismatch")
end
sources = 4.times.map { |i| {path: caches[i], sha256: digests[i]} }.to_a
hy_bytes = HyNomasConditionedFourProbe.admit(hy_fixture, sources, budget, mode)
config = fixture.config
tokens = prepared.tokens.size
unique_ids = prepared.tokens.uniq.sort
gather_sizes = ML::GGUF::Qwen25VLEmbeddingPlan.resident_bytes(config, tokens, unique_ids.size)
prefix_bytes = ML::GGUF::Qwen25VLPrefixPlan.named_bytes(config, tokens, 26, rms_power_fed: true)
crop_sizes = ML::GGUF::Qwen25VLConditioningPlan.resident_bytes(config, tokens, prepared.crop_start)
named_bytes = prefix_bytes + gather_sizes.compact_rows + gather_sizes.remapped_ids + crop_sizes.total + hy_bytes
raise ArgumentError.new("live conditioned four-block exceeds aggregate named-buffer budget") unless named_bytes <= budget
STDERR.puts "admitted #{named_bytes} named bytes; live Qwen26 -> full-row refiner -> four J6560 blocks, valid attention prefix5303"

weights = ML::GGUF::Qwen25VLWeights.from_directory(checkpoint_root)
owned = [] of ML::MetalBuffer
layer_weights = [] of Hash(String, ML::MetalBuffer)
embedding : ML::GGUF::Qwen25VLEmbeddingPlan? = nil
prefix : ML::GGUF::Qwen25VLPrefixPlan? = nil
crop : ML::GGUF::Qwen25VLConditioningPlan? = nil
session : HyNomasConditionedFourProbe::Session? = nil
chain : ML::Metal::MPSCommandChain? = nil
requests = [] of JSON::Any
begin
  raw_rows = fixture.verify_embedding_rows(weights)
  table = ML::MetalBuffer.new(raw_rows.size.to_i64)
  owned << table
  table.write_bytes(raw_rows.to_unsafe, raw_rows.size)
  embedding = ML::GGUF::Qwen25VLEmbeddingPlan.new(config, tokens,
    compact_rows: table, row_indices: prepared.tokens.map { |id| unique_ids.index(id).not_nil! },
    max_named_bytes: gather_sizes.total)
  mask_words = Array(UInt16).new(tokens * tokens) do |i|
    query, key = i // tokens, i % tokens
    key <= query && prepared.attention_mask[key] == 1 ? 0_u16 : 0xff7f_u16
  end
  mask = ML::MetalBuffer.new(mask_words.size.to_i64 * 2)
  owned << mask
  mask.write_bytes(mask_words.to_unsafe.as(UInt8*), mask_words.size * 2)
  positions = ML::MetalBuffer.new(tokens.to_i64 * 4)
  owned << positions
  positions.write(prepared.positions)
  inv_freq = ML::MetalBuffer.new(fixture.inv_freq.size.to_i64 * 4)
  owned << inv_freq
  inv_freq.write(fixture.inv_freq)
  26.times do |i|
    set = weights.upload_block_weights(i, fixture.tensor_sha256s)
    layer_weights << set
    set.each_value { |buffer| owned << buffer }
  end
  prefix = ML::GGUF::Qwen25VLPrefixPlan.new(config, tokens, embedding.output, mask,
    positions, inv_freq, layer_weights, prefix_bytes, rms_power_fed: true)
  raise "not all prefix RMS sites use the GPU-fed exponent" unless prefix.blocks.all?(&.rms_power_fed)
  raw_mask = ML::MetalBuffer.new(crop_sizes.raw_mask)
  owned << raw_mask
  raw_mask.write_bytes(prepared.attention_mask.to_unsafe.as(UInt8*), crop_sizes.raw_mask.to_i)
  crop = ML::GGUF::Qwen25VLConditioningPlan.new(config, tokens, prepared.crop_start,
    prefix.blocks.last.output, raw_mask, crop_sizes.total)
  session = HyNomasConditionedFourProbe::Session.new(hy_fixture, crop.hidden,
    prepared.cropped_attention_mask, sources, hy_bytes, mode)
  count = mode == "positive" ? 2 : 1
  count.times do |request|
    chain = ML::Metal::MPSCommandChain.new
    allocations = ML::MetalBuffer.hy_text_probe_allocations
    reads = ML::MetalBuffer.hy_text_probe_readbacks
    started = Time.instant
    embedding.encode(chain)
    prefix.encode(chain)
    crop.encode(chain)
    session.encode(chain, poison: request > 0)
    encode_seconds = (Time.instant - started).total_seconds
    encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
    early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
    raise "encode allocated Metal buffers or read activations or waited" unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
    early_release_rejections = 0
    if request == 0
      {prefix, crop, session}.each do |resource|
        begin
          resource.release
        rescue ex : ArgumentError
          message = ex.message || ""
          raise ex unless message.includes?("terminal wait") || message.includes?("command chain")
          early_release_rejections += 1
        end
      end
      raise "in-flight producer/crop/consumer release accepted" unless early_release_rejections == 3
    end
    started = Time.instant
    chain.commit_and_wait
    wait_seconds = (Time.instant - started).total_seconds
    raise "expected one terminal wait" unless chain.wait_count == 1
    prefix_comparisons = [fixture.compare(embedding.output, 0)]
    prefix.blocks.each_with_index { |block, i| prefix_comparisons << fixture.compare(block.output, i + 1) }
    crop_comparison = fixture.compare_conditioning(crop.hidden, crop.mask)
    comparisons = session.compare(request.to_i32)
    diagnostic_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
    raise "unexpected post-terminal diagnostic read count" unless diagnostic_reads == 39
    producer_passed = prefix_comparisons.all? { |r| r[:failures] == 0 } && crop_comparison[:failures] == 0 && crop_comparison[:mask_failures] == 0
    output_passed = if mode == "positive"
                      comparisons.all? { |r| r[:finite] && r[:failures] == 0 && r[:first_request_sha256_match] != false }
                    else
                      HyNomasConditionedFourProbe.negative_detected?(mode, comparisons)
                    end
    requests << JSON.parse({request: request, scratch_poisoned: request > 0,
                            poison_buffer_count: request > 0 ? session.poison_count : 0,
                            named_allocations_during_encode: encode_allocations,
                            activation_readbacks_before_terminal_wait: early_reads,
                            diagnostic_readbacks_after_terminal_wait: diagnostic_reads,
                            terminal_wait_count: chain.wait_count,
                            early_release_rejections: early_release_rejections,
                            command_segments: chain.segment_count,
                            encode_seconds: encode_seconds, terminal_wait_seconds: wait_seconds,
                            gpu_elapsed_seconds: chain.gpu_elapsed_seconds,
                            qwen_comparisons: prefix_comparisons, crop_comparison: crop_comparison,
                            comparisons: comparisons, producer_passed: producer_passed,
                            output_or_negative_control_passed: output_passed,
                            passed: producer_passed && output_passed}.to_json)
    STDERR.puts "request #{request + 1}/#{count}: producer=#{producer_passed}, conditioned four-block=#{output_passed}; encode=#{encode_seconds}s, terminal=#{wait_seconds}s"
    chain.release
    chain = nil
    break unless producer_passed && output_passed
  end
  passed = requests.size == count && requests.all? { |r| r["passed"].as_bool }
  result = {schema: "cogni_ml.hy_nomas_conditioned_four_native_comparison", schema_version: 1,
            reference: hy_fixture.path, reference_sha256: hy_fixture.sha256,
            qwen_reference: fixture.manifest_path, qwen_reference_sha256: fixture.manifest_sha256,
            mode: mode, named_bytes: named_bytes, consumer_named_bytes: hy_bytes,
            geometry: hy_fixture.document["geometry"], padding_policy: hy_fixture.document["padding_policy"],
            weight_sources: sources, native_prompt_prepared: true, qwen_layers: 26,
            producer_and_consumer_share_chain: true, direct_gpu_conditioning_handoff: true,
            precision_scope: "BF16 resident parameters; F32 neural math and MPS attention; explicit zero padded query results",
            tolerance: {atol: HyNomasPretrainedSerialProbe::ATOL, rtol: HyNomasPretrainedSerialProbe::RTOL},
            requests: requests, passed: passed}
  HyNomasFourBlockProbe.write_report_new!(report, result.to_pretty_json)
  raise "conditioned four-block comparison/control failed; inspect report" unless passed
  puts "PASS: #{mode}; live native Qwen26/refiner/DiT4; one terminal wait per request"
ensure
  Qwen25PrefixProbe.cleanup_after_drain(chain) do
    session.try(&.release)
    crop.try(&.release)
    prefix.try(&.release)
    embedding.try(&.release)
    chain.try(&.release)
    owned.reverse_each { |buffer| buffer.release if buffer.valid? }
    weights.close
  end
end
