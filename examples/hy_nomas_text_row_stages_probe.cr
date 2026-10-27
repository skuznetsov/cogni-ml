require "option_parser"
require "./support/hy_nomas_probe_instrumentation"
require "./support/hy_nomas_q4_dit_probe"

# Compare padded and valid-prefix block-0 paths after replaying the same
# authenticated raw producers and Q4 block weights. This localizes arithmetic
# drift; it is not a full DiT forward or a timing claim.
module ML::Video
  class HyNomasMPSSerialWorkspace
    # The normal serial owner keeps its plan private. This accessor exists only
    # in this diagnostic example so one registered block reuses that plan.
    def diagnostic_workspace : HyNomasMPSWorkspace
      workspace
    end
  end
end

class HyNomasQ4DiTProbe::Session
  def encode_block0_diagnostic(chain : ML::Metal::MPSCommandChain) : Nil
    raise ArgumentError.new("Full DiT session has been released") if @released
    if previous = @last_chain
      raise ArgumentError.new("Full DiT previous command chain is pending") unless previous.completed? && !previous.same?(chain)
    end

    patch = @patch || raise ArgumentError.new("Full DiT patch plan is unavailable")
    refiner = @refiner || raise ArgumentError.new("Full DiT refiner plan is unavailable")
    serial = @serial || raise ArgumentError.new("Full DiT serial workspace is unavailable")
    final = @final || raise ArgumentError.new("Full DiT final plan is unavailable")
    loader = @loader || raise ArgumentError.new("Full DiT Q4 loader is unavailable")

    # Read-only preflight for producer, refiner, all registered blocks, and the
    # selected Q4 block completes before this method appends any commands.
    patch.validate_for!(chain)
    refiner.validate_for!(chain)
    serial.validate_for!(chain)
    final.validate_for!(chain)
    loader.validate_for!(0, @stage)
    @fixture.unchanged!
    @last_chain = chain

    @prepared.not_nil!.encode(chain.encoder)
    # The active square dispatch overwrites only the compact prefix. Poisoning
    # the whole RMS scratch leaves its inactive tail as a direct row-isolation
    # falsifier for the original-size MPSGraph mean, outside timing.
    if @compact_text
      serial.hy_dit_probe_rms_scratch.each do |buffer|
        chain.encoder.set_pipeline(@poison.not_nil!).set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
          .set_value((buffer.size // 4).to_u32, 1).set_value(0_u32, 2).dispatch_1d((buffer.size // 4).to_i32)
      end
    end
    chain.encoder.set_pipeline(@convert.not_nil!).set_buffer(@inputs["input.qwen_hidden"], 0)
      .set_buffer(@inputs["qwen_bf16"], 1, ML::Metal::BufferAccess::Write)
      .set_value((1280 * 3584).to_u32, 2).dispatch_1d(1280 * 3584)
    patch.encode(chain)
    refiner.encode(chain)
    loader.encode!(chain, 0, @stage)

    text = refiner.text_output
    if @compact_text
      text = serial.compact_text_input || raise ArgumentError.new("Compact text input is unavailable")
      count = ML::Video::HyNomasDualStreamBlock.checked_count(
        serial.block_config.text_rows, serial.block_config.dim,
      )
      ML::Video::MetalOps.copy(chain.encoder, refiner.text_output, text, count)
    end
    serial.blocks[0].encode_mps(chain, patch.output, text, refiner.vec,
      refiner.vec_txt, @inputs["input.rope_cos"], @inputs["input.rope_sin"],
      serial.diagnostic_workspace)
    nil
  end

  def block0_diagnostic_checkpoints : Hash(String, ML::MetalBuffer)
    serial = @serial || raise ArgumentError.new("Full DiT serial workspace is unavailable")
    serial.blocks[0].checkpoints
  end

  def diagnostic_fixture_unchanged! : Nil
    @fixture.unchanged!
  end
end

module HyNomasTextRowStagesProbe
  extend self

  VALID_TEXT_TOKENS  =   23
  PADDED_TEXT_TOKENS = 1280
  DIM                = 2048
  VALID_JOINT_TOKENS = 5280 + VALID_TEXT_TOKENS
  MAX_NAMED_BYTES    = 14_i64 * 1024**3

  # Checkpoint names in dataflow order, from block construction/encoding.
  BLOCK_DATAFLOW = [
    "img.modulation", "txt.modulation",
    "img.attn.norm", "img.attn.modulated", "txt.attn.norm", "txt.attn.modulated",
    "img.attn.q", "img.attn.k", "img.attn.v", "img.attn.q_norm", "img.attn.k_norm",
    "img.attn.q_rope", "img.attn.k_rope",
    "txt.attn.q", "txt.attn.k", "txt.attn.v", "txt.attn.q_norm", "txt.attn.k_norm",
    "joint.query", "joint.key", "joint.value", "joint.attention",
    "img.attn.raw", "txt.attn.raw",
    "img.attn.projected", "img.after_attention",
    "txt.attn.projected", "txt.after_attention",
    "img.ff.norm", "img.ff.modulated", "img.ff.hidden", "img.ff.activated",
    "img.ff.projected", "img.output",
    "txt.ff.norm", "txt.ff.modulated", "txt.ff.hidden", "txt.ff.activated",
    "txt.ff.projected", "txt.output",
  ]

  alias Snapshot = Hash(String, Array(Float32))

  def comparable_elements(name : String, padded_elements : Int32) : Int32
    if name.starts_with?("img.")
      padded_elements
    elsif name.starts_with?("joint.")
      VALID_JOINT_TOKENS * DIM
    elsif name == "txt.modulation"
      # This remains one broadcast modulation row in both configurations.
      padded_elements
    elsif name.starts_with?("txt.")
      unless padded_elements % PADDED_TEXT_TOKENS == 0
        raise ArgumentError.new("Padded text checkpoint has a partial row: #{name}")
      end
      (padded_elements // PADDED_TEXT_TOKENS) * VALID_TEXT_TOKENS
    else
      raise ArgumentError.new("Unknown block-0 checkpoint: #{name}")
    end
  end

  def validate_checkpoint_names!(buffers : Hash(String, ML::MetalBuffer)) : Nil
    unless buffers.keys.sort == BLOCK_DATAFLOW.sort
      raise ArgumentError.new("Block-0 diagnostic checkpoint inventory changed")
    end
  end

  def read_snapshot(session : HyNomasQ4DiTProbe::Session,
                    chain : ML::Metal::MPSCommandChain) : Snapshot
    unless chain.completed? && chain.wait_count == 1
      raise ArgumentError.new("Block-0 snapshots require one terminal wait")
    end
    buffers = session.block0_diagnostic_checkpoints
    validate_checkpoint_names!(buffers)
    BLOCK_DATAFLOW.to_h do |name|
      buffer = buffers[name]
      padded_elements = (buffer.size // sizeof(Float32)).to_i32
      count = comparable_elements(name, padded_elements)
      unless count > 0 && count <= padded_elements
        raise ArgumentError.new("Invalid block-0 comparable extent: #{name}")
      end
      values = Array(Float32).new(count, 0_f32)
      buffer.read_to_slice(Slice.new(values.to_unsafe, values.size))
      {name, values}
    end
  end

  def compare_stage(name : String, padded : Array(Float32),
                    compact : ML::MetalBuffer, storage_rows : Int32) : JSON::Any
    count = padded.size
    compact_elements = (compact.size // sizeof(Float32)).to_i
    expected_elements = if name.starts_with?("joint.")
                          (5280 + storage_rows) * DIM
                        elsif name.starts_with?("txt.") && name != "txt.modulation"
                          (count // VALID_TEXT_TOKENS) * storage_rows
                        else
                          count
                        end
    unless count > 0 && compact_elements == expected_elements
      raise ArgumentError.new("Padded/compact block-0 extent differs at #{name}: #{count} vs #{compact_elements}")
    end

    actual = Array(Float32).new(count, 0_f32)
    compact.read_to_slice(Slice.new(actual.to_unsafe, actual.size))
    finite = true
    failures = 0
    raw_differences = 0
    max_abs = 0_f64
    first_raw = nil.as(JSON::Any?)
    first_failure = nil.as(JSON::Any?)
    count.times do |i|
      expected = padded[i]
      value = actual[i]
      different = expected.unsafe_as(UInt32) != value.unsafe_as(UInt32)
      raw_differences += 1 if different
      if different && first_raw.nil?
        first_raw = JSON.parse({index: i, padded_value: expected, compact_value: value,
                                padded_bits: expected.unsafe_as(UInt32).to_s,
                                compact_bits: value.unsafe_as(UInt32).to_s}.to_json)
      end

      if expected.finite? && value.finite?
        error = (value.to_f64 - expected.to_f64).abs
        limit = HyNomasQ4DiTProbe::ATOL + HyNomasQ4DiTProbe::RTOL * expected.to_f64.abs
        max_abs = Math.max(max_abs, error)
        if error > limit
          failures += 1
          if first_failure.nil?
            first_failure = JSON.parse({index: i, padded_value: expected, compact_value: value,
                                        abs_error: error, tolerance_limit: limit}.to_json)
          end
        end
      else
        finite = false
        failures += 1
        if first_failure.nil?
          first_failure = JSON.parse({index: i, padded_value: expected, compact_value: value,
                                      abs_error: nil, tolerance_limit: nil}.to_json)
        end
      end
    end

    JSON.parse({name: name, elements: count, finite: finite,
                tolerance_failures: failures, raw_bit_differences: raw_differences,
                max_abs_error: finite ? max_abs : nil,
                first_raw_difference: first_raw,
                first_tolerance_failure: first_failure}.to_json)
  end

  def compare_live(session : HyNomasQ4DiTProbe::Session,
                   padded : Snapshot, storage_rows : Int32) : Array(JSON::Any)
    buffers = session.block0_diagnostic_checkpoints
    validate_checkpoint_names!(buffers)
    unless padded.keys.sort == BLOCK_DATAFLOW.sort
      raise ArgumentError.new("Padded block-0 snapshot inventory changed")
    end
    BLOCK_DATAFLOW.map { |name| compare_stage(name, padded[name], buffers[name], storage_rows) }
  end

  def run_padded(fixture : HyNomasQ4DiTProbe::Fixture, index_path : String,
                 index_sha : String, index : ML::Video::HyNomasQ4StackIndex::Index,
                 budget : Int64, expected_bytes : Int64) : Tuple(Snapshot, JSON::Any)
    session = nil.as(HyNomasQ4DiTProbe::Session?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      session = HyNomasQ4DiTProbe::Session.new(fixture, index_path, index_sha,
        index, "q4", budget, compact_text: false)
      raise "Padded session differs from preflight" unless session.named_bytes == expected_bytes
      chain = ML::Metal::MPSCommandChain.new
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      reads = ML::MetalBuffer.hy_text_probe_readbacks
      session.encode_block0_diagnostic(chain)
      encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
      early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
      unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
        raise "Padded block-0 encode allocated/read back/waited"
      end
      chain.commit_and_wait
      snapshot = read_snapshot(session, chain)
      session.diagnostic_fixture_unchanged!
      readbacks = ML::MetalBuffer.hy_text_probe_readbacks - reads
      unless readbacks == BLOCK_DATAFLOW.size
        raise "Padded block-0 checkpoint read count changed: #{readbacks}"
      end
      record = JSON.parse({mode: "padded", named_bytes: session.named_bytes,
                           encode_metal_allocations: encode_allocations,
                           preterminal_readbacks: early_reads,
                           postterminal_checkpoint_readbacks: readbacks,
                           terminal_wait_count: chain.wait_count,
                           command_segments: chain.segment_count}.to_json)
      {snapshot, record}
    ensure
      if current = chain
        current.commit_and_wait unless current.completed?
      end
      session.try(&.release)
      chain.try(&.release)
    end
  end

  def run_compact(fixture : HyNomasQ4DiTProbe::Fixture, index_path : String,
                  index_sha : String, index : ML::Video::HyNomasQ4StackIndex::Index,
                  budget : Int64, expected_bytes : Int64,
                  padded : Snapshot, storage_rows : Int32) : Tuple(Array(JSON::Any), JSON::Any)
    session = nil.as(HyNomasQ4DiTProbe::Session?)
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      session = HyNomasQ4DiTProbe::Session.new(fixture, index_path, index_sha,
        index, "q4", budget, compact_text: true, compact_text_tokens: storage_rows)
      raise "Compact session differs from preflight" unless session.named_bytes == expected_bytes
      chain = ML::Metal::MPSCommandChain.new
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      reads = ML::MetalBuffer.hy_text_probe_readbacks
      session.encode_block0_diagnostic(chain)
      encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations
      early_reads = ML::MetalBuffer.hy_text_probe_readbacks - reads
      unless encode_allocations == 0 && early_reads == 0 && chain.wait_count == 0
        raise "Compact block-0 encode allocated/read back/waited"
      end
      chain.commit_and_wait
      comparisons = compare_live(session, padded, storage_rows)
      session.diagnostic_fixture_unchanged!
      readbacks = ML::MetalBuffer.hy_text_probe_readbacks - reads
      unless readbacks == BLOCK_DATAFLOW.size && chain.wait_count == 1
        raise "Compact block-0 checkpoint reads or terminal wait changed"
      end
      record = JSON.parse({mode: "compact", named_bytes: session.named_bytes,
                           rms_scratch_poisoned: true,
                           encode_metal_allocations: encode_allocations,
                           preterminal_readbacks: early_reads,
                           postterminal_checkpoint_readbacks: readbacks,
                           terminal_wait_count: chain.wait_count,
                           command_segments: chain.segment_count}.to_json)
      {comparisons, record}
    ensure
      if current = chain
        current.commit_and_wait unless current.completed?
      end
      session.try(&.release)
      chain.try(&.release)
    end
  end
end

reference, reference_sha, index_path, index_sha, report = "", "", "", "", ""
mode = "q4"
budget = 0_i64
compact_text_tokens = 23
OptionParser.parse do |parser|
  parser.banner = "Padded/valid-prefix block-0 replay using authenticated Q4 weights."
  parser.on("--reference PATH", "Pinned upstream full DiT fixture") { |v| reference = v }
  parser.on("--reference-sha256 SHA", "Trusted reference SHA256") { |v| reference_sha = v }
  parser.on("--cache-index PATH", "Authenticated ordered 54-block cache index") { |v| index_path = v }
  parser.on("--cache-index-sha256 SHA", "Trusted cache index SHA256") { |v| index_sha = v }
  parser.on("--precision MODE", "Only q4 is admitted by this diagnostic") { |v| mode = v }
  parser.on("--max-named-bytes N", "Per-session aggregate named residency budget") { |v| budget = v.to_i64 }
  parser.on("--compact-text-tokens N", "Aligned storage rows; attention retains 23 valid tokens") { |v| compact_text_tokens = v.to_i32 }
  parser.on("--report PATH", "New block-0 stage comparison report") { |v| report = v }
end

unless ARGV.empty? && {reference, reference_sha, index_path, index_sha, report}.all? { |v| !v.empty? } &&
       mode == "q4" && 0 < budget <= HyNomasTextRowStagesProbe::MAX_NAMED_BYTES
  raise ArgumentError.new("Expected authenticated q4 inputs, new report, and per-session budget<=14 GiB")
end
raise ArgumentError.new("Refusing to replace block-0 diagnostic report") if File.info?(report, false)

admitted_index = ML::Video::HyNomasQ4StackIndex.load!(index_path, index_sha,
  HyNomasQ4DiTIO::SOURCE_SHA, HyNomasQ4DiTIO::CONFIG)
fixture = HyNomasQ4DiTProbe::Fixture.new(reference, reference_sha, "q4", admitted_index)
padded_bytes = HyNomasQ4DiTProbe.q4_named_bytes(admitted_index)
compact_bytes = HyNomasQ4DiTProbe.q4_named_bytes(admitted_index, compact_text: true, compact_text_tokens: compact_text_tokens)
unless padded_bytes <= budget && compact_bytes <= budget
  raise ArgumentError.new("Each sequential full session must fit the admitted per-session budget")
end

padded_snapshot, padded_run = HyNomasTextRowStagesProbe.run_padded(fixture,
  index_path, index_sha, admitted_index, budget, padded_bytes)
comparisons, compact_run = HyNomasTextRowStagesProbe.run_compact(fixture,
  index_path, index_sha, admitted_index, budget, compact_bytes, padded_snapshot, compact_text_tokens)
first_raw = comparisons.find { |entry| entry["raw_bit_differences"].as_i > 0 }
first_failure = comparisons.find do |entry|
  entry["tolerance_failures"].as_i > 0 || !entry["finite"].as_bool
end
fixture.unchanged!

passed = comparisons.all? do |entry|
  entry["finite"].as_bool && entry["tolerance_failures"].as_i == 0
end
result = {
  schema:                         "cogni_ml.hy_nomas_block0_text_row_stages_diagnostic",
  schema_version:                 1,
  reference:                      fixture.path,
  reference_sha256:               fixture.sha256,
  cache_index:                    index_path,
  cache_index_sha256:             index_sha,
  cache_payload_bytes:            admitted_index.caches.sum(&.payload_bytes),
  source_checkpoint_sha256:       HyNomasQ4DiTIO::SOURCE_SHA,
  precision_mode:                 "q4",
  block_index:                    0,
  padded_text_tokens:             HyNomasTextRowStagesProbe::PADDED_TEXT_TOKENS,
  valid_text_tokens:              HyNomasTextRowStagesProbe::VALID_TEXT_TOKENS,
  compact_text_tokens:            compact_text_tokens,
  valid_joint_tokens:             HyNomasTextRowStagesProbe::VALID_JOINT_TOKENS,
  per_session_named_budget_bytes: budget,
  padded_session_named_bytes:     padded_bytes,
  compact_session_named_bytes:    compact_bytes,
  execution:                      "sequential padded and compact block-0; identical authenticated raw inputs, full patch/refiner replay, block-0 Q4 load; one terminal wait each",
  raw_inputs_only:                true,
  expected_activations_uploaded:  false,
  diffusion_generation:           false,
  tolerance:                      {atol: HyNomasQ4DiTProbe::ATOL, rtol: HyNomasQ4DiTProbe::RTOL},
  padded_run:                     padded_run,
  compact_run:                    compact_run,
  first_raw_difference_stage:     first_raw.try { |entry| entry["name"].as_s },
  first_tolerance_failure_stage:  first_failure.try { |entry| entry["name"].as_s },
  first_raw_difference:           first_raw.try { |entry| entry["first_raw_difference"] },
  first_tolerance_failure:        first_failure.try { |entry| entry["first_tolerance_failure"] },
  comparisons:                    comparisons,
  passed:                         passed,
}
HyNomasQ4DiTIO.write_new!(report, result.to_pretty_json + "\n")
puts({report: report, passed: passed,
      first_raw_difference_stage: result[:first_raw_difference_stage],
      first_tolerance_failure_stage: result[:first_tolerance_failure_stage]}.to_json)
raise "Padded/compact block-0 stage comparison failed unchanged tolerance" unless passed
