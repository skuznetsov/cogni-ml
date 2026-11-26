require "json"
require "option_parser"
require "./support/hy_nomas_refiner_transfer_probe"

include HyNomasRefinerTransferProbe

reference = ""
reference_sha256 = ""
report = ""
max_named_bytes = 3_i64 * 1024 * 1024 * 1024
admit_only = false

OptionParser.parse do |parser|
  parser.banner = "Authenticated A/A/B Hy-Nomas refiner transfer probe; refiner only, no DiT."
  parser.on("--reference PATH", "Source-produced refiner transfer manifest") { |value| reference = value }
  parser.on("--reference-sha256 SHA256", "Trusted source reference manifest digest") { |value| reference_sha256 = value }
  parser.on("--report PATH", "New native report in the reference directory") { |value| report = value }
  parser.on("--max-named-bytes N", "Aggregate named-buffer budget; default 3 GiB") { |value| max_named_bytes = value.to_i64 }
  parser.on("--admit-only", "Run CPU self-tests and authenticate the fixture; allocate no Metal resources") { admit_only = true }
end

raise ArgumentError.new("reference and trusted reference SHA-256 are required") if reference.empty? || reference_sha256.empty?
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
raise ArgumentError.new("named-buffer budget must be positive") unless max_named_bytes > 0
raise ArgumentError.new("--report is required unless --admit-only is selected") if report.empty? && !admit_only
raise ArgumentError.new("--report is not accepted with --admit-only") if admit_only && !report.empty?

HyNomasRefinerTransferProbe.admission_self_test!
fixture = HyNomasRefinerTransferProbe::Fixture.new(reference, reference_sha256)
if admit_only
  admission_json = JSON.build do |json|
    json.object do
      json.field "admitted", true
      json.field "reference", fixture.path
      json.field "reference_sha256", fixture.sha256
      json.field "source_AA_within_tolerance", true
      json.field "raw_prefix_rows", "[0,132) bit-identical before padding normalization"
      json.field "action2_changed_bytes", fixture.input_action2_changed_bytes
      json.field "action2_changed_rows", fixture.input_action2_changed_rows
      json.field "raw_padding_changed_bytes", fixture.raw_padding_changed_bytes
    end
  end
  puts admission_json
  exit 0
end

reference_dir = File.dirname(fixture.path)
unless Path[report].absolute? && File.realpath(File.dirname(report)) == reference_dir
  raise ArgumentError.new("native report must be an absolute fresh path beside the reference manifest")
end
report = File.join(reference_dir, Path[report].basename)
raise ArgumentError.new("refusing to replace native report") if File.exists?(report) || File.symlink?(report)

checkpoint_file : ML::Video::CheckpointFile? = nil
weight_arena : ML::Video::Arena? = nil
input_arena : ML::Video::Arena? = nil
snapshot_arena : ML::Video::Arena? = nil
prepared : ML::Video::PreparedWeights? = nil
refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
chain : ML::Metal::MPSCommandChain? = nil
wait_attempted = false

begin
  checkpoint_file = ML::Video::CheckpointFile.new(fixture.text_weights_path,
    expected_info: fixture.text_weights_info)
  weight_plan = ML::Video::WeightPlan.new([checkpoint_file.not_nil!],
    ML::Video::HyNomasTextRefinerMPSPlan.checkpoint_bindings,
    max_named_bytes, resident_dtype: ML::Video::WeightDType::BF16,
    require_all: true, allow_f32_to_bf16: true)

  refiner_bytes = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(fixture.mask,
    retain_padding: true).total
  input_bytes = 3_i64 * HyNomasRefinerTransferProbe::INPUT_BYTES
  snapshot_elements = HyNomasRefinerTransferProbe::OUTPUT_CHECKPOINTS.sum do |checkpoint|
    HyNomasRefinerTransferProbe.tensor_elements(checkpoint).to_i64
  end
  snapshot_bytes = 3_i64 * snapshot_elements * sizeof(Float32)
  total_named_bytes = weight_plan.peak_bytes + refiner_bytes + input_bytes + snapshot_bytes
  raise ArgumentError.new("probe named buffers exceed --max-named-bytes") if total_named_bytes > max_named_bytes

  # Metal setup starts only after full CPU fixture/source/output admission and
  # aggregate named-byte preflight have completed.
  input_arena = ML::Video::Arena.new
  weight_arena = ML::Video::Arena.new
  snapshot_arena = ML::Video::Arena.new
  qwen = input_arena.not_nil!.alloc_weight("qwen_working", 1280 * 3584,
    ML::Video::WeightDType::BF16)
  input_a = input_arena.not_nil!.alloc_weight("source_a", 1280 * 3584,
    ML::Video::WeightDType::BF16)
  input_b = input_arena.not_nil!.alloc_weight("source_b", 1280 * 3584,
    ML::Video::WeightDType::BF16)
  input_a.write_bytes(fixture.normalized_input("A0").to_unsafe, HyNomasRefinerTransferProbe::INPUT_BYTES.to_i32)
  input_b.write_bytes(fixture.normalized_input("B").to_unsafe, HyNomasRefinerTransferProbe::INPUT_BYTES.to_i32)

  prepared = ML::Video::PreparedWeights.new(weight_plan, weight_arena.not_nil!)
  refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(qwen, fixture.mask,
    fixture.timestep, weight_arena.not_nil!, refiner_bytes,
    ML::Video::HyNomasAttentionPrecision::F32Control,
    retain_padding: true,
    prelude_input_bf16_control: true,
    prelude_linear_bf16_control: true,
    prelude_silu_bf16_control: true,
    prelude_second_linear_bf16_control: true,
    prelude_add_bf16_control: true,
    block_boundaries_bf16_control: true,
    qkv_bf16_control: true,
    source_attention_control: true,
    source_projection_residual_control: true,
    source_norm2_control: true,
    source_fc1_control: true,
    source_mlp_silu_control: true,
    source_fc2_residual_control: true,
    source_mean_autocast: true)
  raise "V224 refiner storage was compacted" unless refiner.not_nil!.processed_tokens == 1280
  raise "refiner named-byte plan does not match admission" unless refiner.not_nil!.resident_bytes == refiner_bytes

  checkpoints = refiner.not_nil!.checkpoints
  destinations = {} of String => Hash(String, ML::MetalBuffer)
  observations = {} of String => Hash(String, Array(Float32))
  CASES.each do |case_name|
    case_destinations = {} of String => ML::MetalBuffer
    case_observations = {} of String => Array(Float32)
    OUTPUT_CHECKPOINTS.each do |checkpoint|
      count = tensor_elements(checkpoint)
      snapshot = snapshot_arena.not_nil!.alloc("#{case_name}.#{checkpoint}", count)
      case_destinations[checkpoint] = snapshot
      case_observations[checkpoint] = Array(Float32).new(count, 0.0_f32)
    end
    destinations[case_name] = case_destinations
    observations[case_name] = case_observations
  end

  copy_plan = BF16CopyPlan.new
  capture_plan = ML::Video::HyNomasContinuationHistoryKernels::AppendPlan.new
  active_chain = ML::Metal::MPSCommandChain.new
  chain = active_chain
  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
  encode_started = Time.instant
  prepared.not_nil!.encode(active_chain.encoder)
  encode_transfer_case(active_chain, input_a, qwen, refiner.not_nil!, checkpoints,
    destinations["A0"], copy_plan, capture_plan)
  encode_transfer_case(active_chain, input_a, qwen, refiner.not_nil!, checkpoints,
    destinations["A1"], copy_plan, capture_plan)
  encode_transfer_case(active_chain, input_b, qwen, refiner.not_nil!, checkpoints,
    destinations["B"], copy_plan, capture_plan)
  encode_seconds = (Time.instant - encode_started).total_seconds
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  raise "probe encode allocated named GPU buffers" unless encode_allocations == 0
  raise "probe encode performed an activation readback" unless encode_readbacks == 0
  raise "probe encode performed an intermediate wait" unless active_chain.wait_count == 0

  wait_attempted = true
  wait_started = Time.instant
  active_chain.commit_and_wait
  wait_seconds = (Time.instant - wait_started).total_seconds
  raise "expected exactly one terminal chain drain" unless active_chain.completed? && active_chain.wait_count == 1

  CASES.each do |case_name|
    OUTPUT_CHECKPOINTS.each do |checkpoint|
      values = observations[case_name][checkpoint]
      destinations[case_name][checkpoint].read_to_slice(Slice.new(values.to_unsafe, values.size))
    end
  end
  postwait_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  raise "expected exactly 15 post-terminal checkpoint readbacks" unless postwait_readbacks == 15

  parity = [] of NamedTuple(case_name: String, checkpoint: String, elements: Int32,
    failures: Int32, max_abs: Float64, raw_bit_differences: Int32,
    first_mismatch: Int32?, first_actual: Float32?, first_expected: Float32?,
    finite: Bool, passed: Bool)
  parity_failed = false
  CASES.each do |case_name|
    OUTPUT_CHECKPOINTS.each do |checkpoint|
      result = compare_values(observations[case_name][checkpoint], fixture.expected_values("#{case_name}.#{checkpoint}"))
      passed = result[:failures] == 0 && result[:finite]
      parity_failed ||= !passed
      parity << {case_name: case_name, checkpoint: checkpoint,
                 elements: tensor_elements(checkpoint), failures: result[:failures],
                 max_abs: result[:max_abs], raw_bit_differences: result[:raw_bit_differences],
                 first_mismatch: result[:first_mismatch], first_actual: result[:first_actual],
                 first_expected: result[:first_expected], finite: result[:finite], passed: passed}
    end
  end

  comparisons = {} of String => Hash(String, DeltaStats)
  aa_within_tolerance = true
  {"A0_vs_A1" => {"A0", "A1"}, "A1_vs_B" => {"A1", "B"}}.each do |comparison_name, pair|
    baseline_name = pair[0]
    changed_name = pair[1]
    stats = {} of String => DeltaStats
    OUTPUT_CHECKPOINTS.each do |checkpoint|
      left = observations[baseline_name][checkpoint]
      right = observations[changed_name][checkpoint]
      stats[checkpoint] = delta_stats(left, right)
      if comparison_name == "A0_vs_A1"
        aa_check = compare_values(left, right)
        aa_within_tolerance &&= aa_check[:failures] == 0 && aa_check[:finite]
      end
      if checkpoint == "block0.output" || checkpoint == "block1.output"
        SPANS.each do |span_name, range|
          stats["#{checkpoint}.#{span_name}"] = delta_stats(span(left, range[0], range[1]),
            span(right, range[0], range[1]))
        end
      end
    end
    comparisons[comparison_name] = stats
  end

  report_json = JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", "hy-nomas-refiner-transfer-native-v1"
      json.field "schema_version", 1
      json.field "reference", fixture.path
      json.field "reference_sha256", fixture.sha256
      json.field "text_weights_sha256", fixture.text_weights_sha256
      json.field "input_normalized_sha256" do
        json.object do
          CASES.each { |case_name| json.field(case_name, fixture.inputs[case_name].normalized_tensor_sha256) }
        end
      end
      json.field "geometry", {"input_shape" => [1, 1280, 3584], "valid_rows" => VALID_ROWS,
                              "action_spans" => {"global" => [0, 49], "action1" => [49, 132], "action2" => [132, 224]},
                              "checkpoint_rows" => "valid prefix [0,224) only"}
      json.field "precision", {"resident_weights" => "BF16; authenticated F32 source subset converted by native RNE path",
                               "compute" => "F32", "attention" => "F32Control", "source_bf16_controls" => true,
                               "timestep" => fixture.timestep, "vec_semantics" => "time_in output; timestep-only"}
      json.field "resource_accounting" do
        json.object do
          json.field "weight_plan_peak_bytes", weight_plan.peak_bytes
          json.field "refiner_named_bytes", refiner_bytes
          json.field "input_buffers_bytes", input_bytes
          json.field "snapshot_buffers_bytes", snapshot_bytes
          json.field "aggregate_named_bytes", total_named_bytes
          json.field "max_named_bytes", max_named_bytes
          json.field "opaque_mps_resources_included", false
        end
      end
      json.field "execution" do
        json.object do
          json.field "scope", "refiner only; no DiT"
          json.field "case_order", CASES
          json.field "prepared_weights_encodes", 1
          json.field "command_chains", 1
          json.field "terminal_waits", active_chain.wait_count
          json.field "named_allocations_during_encode", encode_allocations
          json.field "readbacks_before_terminal_wait", encode_readbacks
          json.field "postwait_checkpoint_readbacks", postwait_readbacks
          json.field "encode_seconds", encode_seconds
          json.field "terminal_wait_seconds", wait_seconds
          json.field "gpu_elapsed_seconds", active_chain.gpu_elapsed_seconds
        end
      end
      json.field "parity_tolerance", {"atol" => ATOL, "rtol" => RTOL}
      json.field "source_parity" do
        json.array do
          parity.each do |item|
            json.object do
              json.field "case", item[:case_name]
              json.field "checkpoint", item[:checkpoint]
              json.field "elements", item[:elements]
              json.field "failures", item[:failures]
              json.field "max_abs", item[:max_abs]
              json.field "raw_bit_differences", item[:raw_bit_differences]
              json.field "first_mismatch", item[:first_mismatch]
              json.field "first_actual", item[:first_actual]
              json.field "first_expected", item[:first_expected]
              json.field "finite", item[:finite]
              json.field "passed", item[:passed]
            end
          end
        end
      end
      json.field "A0_A1", {"exact" => comparisons["A0_vs_A1"]["block0.output"].exact_zero &&
                                      comparisons["A0_vs_A1"]["block1.output"].exact_zero && comparisons["A0_vs_A1"]["c"].exact_zero &&
                                      comparisons["A0_vs_A1"]["masked_mean"].exact_zero && comparisons["A0_vs_A1"]["vec"].exact_zero,
                           "within_atol_rtol" => aa_within_tolerance, "deltas" => comparisons["A0_vs_A1"].transform_values { |stats| delta_hash(stats) }}
      json.field "A1_B", {"deltas" => comparisons["A1_vs_B"].transform_values { |stats| delta_hash(stats) }}
      json.field "passed", !parity_failed && aa_within_tolerance
    end
  end
  write_fresh_output(report, report_json)
  puts report_json
  raise "native source parity or A/A stability failed; report preserved at #{report}" if parity_failed || !aa_within_tolerance
ensure
  if active_chain = chain
    unless wait_attempted || active_chain.completed?
      wait_attempted = true
      begin
        active_chain.commit_and_wait
      rescue
      end
    end
    if active_chain.completed?
      refiner.try(&.release)
      prepared.try(&.release)
      input_arena.try(&.release)
      snapshot_arena.try(&.release)
      weight_arena.try(&.release)
      checkpoint_file.try(&.close)
      active_chain.release
    end
  else
    refiner.try(&.release)
    prepared.try(&.release)
    input_arena.try(&.release)
    snapshot_arena.try(&.release)
    weight_arena.try(&.release)
    checkpoint_file.try(&.close)
  end
end

def encode_transfer_case(chain : ML::Metal::MPSCommandChain,
                         source : ML::MetalBuffer, qwen : ML::MetalBuffer,
                         refiner : ML::Video::HyNomasTextRefinerMPSPlan,
                         checkpoints : Hash(String, ML::MetalBuffer),
                         destinations : Hash(String, ML::MetalBuffer),
                         copy_plan : HyNomasRefinerTransferProbe::BF16CopyPlan,
                         capture_plan : ML::Video::HyNomasContinuationHistoryKernels::AppendPlan) : Nil
  copy_plan.encode(chain.encoder, source, qwen, 1280 * 3584)
  refiner.encode(chain)
  HyNomasRefinerTransferProbe::OUTPUT_CHECKPOINTS.each do |checkpoint|
    count = HyNomasRefinerTransferProbe.tensor_elements(checkpoint)
    capture_plan.encode(chain.encoder, checkpoints[checkpoint], destinations[checkpoint], count, 0)
  end
end

def delta_hash(stats : HyNomasRefinerTransferProbe::DeltaStats) : Hash(String, JSON::Any)
  JSON.parse(JSON.build do |json|
    json.object do
      json.field "elements", stats.elements
      json.field "changed_values", stats.changed_values
      json.field "max_abs", stats.max_abs
      json.field "rms_delta", stats.rms_delta
      json.field "baseline_rms", stats.baseline_rms
      json.field "relative_rms_delta", stats.relative_rms_delta
      json.field "exact_zero", stats.exact_zero
      json.field "finite", stats.finite
    end
  end).as_h
end

def write_fresh_output(path : String, content : String) : Nil
  fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
  raise IO::Error.new("native report must be fresh and writable: #{path}") if fd < 0
  io = IO::FileDescriptor.new(fd)
  begin
    io.puts(content)
  ensure
    io.close
  end
end
