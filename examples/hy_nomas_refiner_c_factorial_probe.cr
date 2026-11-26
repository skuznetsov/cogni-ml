require "json"
require "option_parser"
require "./support/hy_nomas_refiner_c_factorial_probe"

include HyNomasRefinerCFactorialProbe

reference = ""
reference_sha256 = ""
report = ""
max_named_bytes = 6_i64 * 1024 * 1024 * 1024
admit_only = false

OptionParser.parse do |parser|
  parser.banner = "Authenticated five-case Hy-Nomas c-factorial probe; refiner only, no DiT."
  parser.on("--reference PATH", "Source-produced factorial reference manifest") { |value| reference = value }
  parser.on("--reference-sha256 SHA256", "Trusted source factorial manifest digest") { |value| reference_sha256 = value }
  parser.on("--report PATH", "New native report beside the reference manifest") { |value| report = value }
  parser.on("--max-named-bytes N", "Aggregate named-buffer budget; default 6 GiB") { |value| max_named_bytes = value.to_i64 }
  parser.on("--admit-only", "Authenticate parent and child references without Metal allocation") { admit_only = true }
end

raise ArgumentError.new("reference and trusted reference SHA-256 are required") if reference.empty? || reference_sha256.empty?
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
raise ArgumentError.new("named-buffer budget must be positive") unless max_named_bytes > 0
raise ArgumentError.new("--report is required unless --admit-only is selected") if report.empty? && !admit_only
raise ArgumentError.new("--report is not accepted with --admit-only") if admit_only && !report.empty?

HyNomasRefinerCFactorialProbe.admission_self_test!
fixture = HyNomasRefinerCFactorialProbe::Fixture.new(reference, reference_sha256)
if admit_only
  admission_json = JSON.build do |json|
    json.object do
      json.field "admitted", true
      json.field "reference", fixture.path
      json.field "reference_sha256", fixture.sha256
      json.field "parent_reference", fixture.parent.path
      json.field "parent_reference_sha256", fixture.parent.sha256
      json.field "factor_cases", CASES
      json.field "output_tensors", fixture.output_tensors.size
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

parent = fixture.parent
checkpoint_file : ML::Video::CheckpointFile? = nil
weight_arena : ML::Video::Arena? = nil
input_arena : ML::Video::Arena? = nil
snapshot_arena : ML::Video::Arena? = nil
prepared : ML::Video::PreparedWeights? = nil
refiner : ML::Video::HyNomasTextRefinerMPSPlan? = nil
chain : ML::Metal::MPSCommandChain? = nil
wait_attempted = false

begin
  checkpoint_file = ML::Video::CheckpointFile.new(parent.text_weights_path,
    expected_info: parent.text_weights_info)
  weight_plan = ML::Video::WeightPlan.new([checkpoint_file.not_nil!],
    ML::Video::HyNomasTextRefinerMPSPlan.checkpoint_bindings,
    max_named_bytes, resident_dtype: ML::Video::WeightDType::BF16,
    require_all: true, allow_f32_to_bf16: true)

  refiner_bytes = ML::Video::HyNomasTextRefinerMPSPlan.named_bytes(parent.mask,
    retain_padding: true).total
  input_bytes = 3_i64 * HyNomasRefinerCFactorialProbe::INPUT_BYTES
  snapshot_elements = HyNomasRefinerCFactorialProbe::OUTPUT_CHECKPOINTS.sum do |checkpoint|
    HyNomasRefinerTransferProbe.tensor_elements(checkpoint).to_i64
  end
  snapshot_bytes = HyNomasRefinerCFactorialProbe::CASES.size.to_i64 * snapshot_elements * sizeof(Float32)
  total_named_bytes = weight_plan.peak_bytes + refiner_bytes + input_bytes + snapshot_bytes
  raise ArgumentError.new("factorial probe named buffers exceed --max-named-bytes") if total_named_bytes > max_named_bytes

  input_arena = ML::Video::Arena.new
  weight_arena = ML::Video::Arena.new
  snapshot_arena = ML::Video::Arena.new
  qwen = input_arena.not_nil!.alloc_weight("qwen_working", 1280 * 3584,
    ML::Video::WeightDType::BF16)
  input_a = input_arena.not_nil!.alloc_weight("source_a", 1280 * 3584,
    ML::Video::WeightDType::BF16)
  input_b = input_arena.not_nil!.alloc_weight("source_b", 1280 * 3584,
    ML::Video::WeightDType::BF16)
  input_a.write_bytes(parent.normalized_input("A0").to_unsafe, HyNomasRefinerCFactorialProbe::INPUT_BYTES.to_i32)
  input_b.write_bytes(parent.normalized_input("B").to_unsafe, HyNomasRefinerCFactorialProbe::INPUT_BYTES.to_i32)

  prepared = ML::Video::PreparedWeights.new(weight_plan, weight_arena.not_nil!)
  refiner = ML::Video::HyNomasTextRefinerMPSPlan.new(qwen, parent.mask,
    parent.timestep, weight_arena.not_nil!, refiner_bytes,
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
      count = HyNomasRefinerTransferProbe.tensor_elements(checkpoint)
      case_destinations[checkpoint] = snapshot_arena.not_nil!.alloc("#{case_name}.#{checkpoint}", count)
      case_observations[checkpoint] = Array(Float32).new(count, 0.0_f32)
    end
    destinations[case_name] = case_destinations
    observations[case_name] = case_observations
  end

  capture_plan = ML::Video::HyNomasContinuationHistoryKernels::AppendPlan.new
  input_copy_plan = HyNomasRefinerTransferProbe::BF16CopyPlan.new
  active_chain = ML::Metal::MPSCommandChain.new
  chain = active_chain
  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
  prepared.not_nil!.encode(active_chain.encoder)
  CASES.each do |case_name|
    source, c_override = case case_name
                         when "A0"  then {input_a, nil.as(ML::MetalBuffer?)}
                         when "A1"  then {input_a, destinations["A0"]["c"]}
                         when "B"   then {input_b, nil.as(ML::MetalBuffer?)}
                         when "X10" then {input_b, destinations["A0"]["c"]}
                         when "X01" then {input_a, destinations["B"]["c"]}
                         else            raise ArgumentError.new("unknown factorial case #{case_name}")
                         end
    encode_factor_case(active_chain, source, qwen, c_override, refiner.not_nil!, checkpoints,
      destinations[case_name], input_copy_plan, capture_plan)
  end
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  raise "factorial encode allocated named GPU buffers" unless encode_allocations == 0
  raise "factorial encode performed activation readbacks" unless encode_readbacks == 0
  raise "factorial encode performed an intermediate wait" unless active_chain.wait_count == 0

  wait_attempted = true
  active_chain.commit_and_wait
  raise "expected exactly one terminal chain drain" unless active_chain.completed? && active_chain.wait_count == 1

  CASES.each do |case_name|
    OUTPUT_CHECKPOINTS.each do |checkpoint|
      values = observations[case_name][checkpoint]
      destinations[case_name][checkpoint].read_to_slice(Slice.new(values.to_unsafe, values.size))
    end
  end
  postwait_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  raise "expected exactly 25 post-terminal checkpoint readbacks" unless postwait_readbacks == 25

  native_grouped = HyNomasRefinerCFactorialProbe.group_outputs(
    CASES.each_with_object({} of String => Array(Float32)) do |case_name, flat|
      OUTPUT_CHECKPOINTS.each { |checkpoint| flat["#{case_name}.#{checkpoint}"] = observations[case_name][checkpoint] }
    end)
  invariant_records = HyNomasRefinerCFactorialProbe.case_invariant_records(native_grouped)
  invariants_passed = invariant_records.all?(&.[:exact])

  parity = [] of NamedTuple(case_name: String, checkpoint: String, elements: Int32,
    failures: Int32, max_abs: Float64, raw_bit_differences: Int32,
    first_mismatch: Int32?, first_actual: Float32?, first_expected: Float32?,
    finite: Bool, passed: Bool)
  parity_failed = false
  CASES.each do |case_name|
    OUTPUT_CHECKPOINTS.each do |checkpoint|
      result = HyNomasRefinerTransferProbe.compare_values(observations[case_name][checkpoint],
        fixture.expected_values("#{case_name}.#{checkpoint}"))
      passed = result[:failures] == 0 && result[:finite]
      parity_failed ||= !passed
      parity << {case_name: case_name, checkpoint: checkpoint,
                 elements: HyNomasRefinerTransferProbe.tensor_elements(checkpoint), failures: result[:failures],
                 max_abs: result[:max_abs], raw_bit_differences: result[:raw_bit_differences],
                 first_mismatch: result[:first_mismatch], first_actual: result[:first_actual],
                 first_expected: result[:first_expected], finite: result[:finite], passed: passed}
    end
  end
  effects = factorial_effects(native_grouped)

  report_json = JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", "hy-nomas-refiner-c-factorial-native-v1"
      json.field "schema_version", 1
      json.field "reference", fixture.path
      json.field "reference_sha256", fixture.sha256
      json.field "parent_reference", parent.path
      json.field "parent_reference_sha256", parent.sha256
      json.field "text_weights_sha256", parent.text_weights_sha256
      json.field "geometry", {"input_shape" => [1, 1280, 3584], "valid_rows" => VALID_ROWS,
                              "action_spans" => {"global" => [0, 49], "action1" => [49, 132], "action2" => [132, 224]},
                              "checkpoint_rows" => "valid prefix [0,224) only"}
      json.field "case_factors" do
        json.object do
          CASES.each do |case_name|
            factor = FACTORS[case_name]
            json.field case_name, {"text_case" => factor[:text_case], "c_case" => factor[:c_case],
                                   "override_c" => factor[:override_c]}
          end
        end
      end
      json.field "precision", {"resident_weights" => "BF16; authenticated F32 source subset converted by native RNE path",
                               "compute" => "F32", "attention" => "F32Control", "source_bf16_controls" => true,
                               "timestep" => parent.timestep, "vec_semantics" => "time_in output; timestep-only"}
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
        end
      end
      json.field "factor_invariants" do
        json.array do
          invariant_records.each do |item|
            json.object do
              json.field "name", item[:name]
              json.field "left_case", item[:left_case]
              json.field "right_case", item[:right_case]
              json.field "checkpoint", item[:checkpoint]
              json.field "elements", item[:elements]
              json.field "raw_bit_differences", item[:raw_bit_differences]
              json.field "finite", item[:finite]
              json.field "exact", item[:exact]
            end
          end
        end
      end
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
      json.field "factorial_effects" do
        json.object do
          effects.each do |effect_name, values|
            json.field effect_name do
              json.object do
                values.each do |name, stats|
                  json.field name do
                    json.object do
                      json.field "elements", stats.elements
                      json.field "changed_values", stats.changed_values
                      json.field "mean_effect", stats.mean_effect
                      json.field "max_abs", stats.max_abs
                      json.field "rms_effect", stats.rms_effect
                      json.field "baseline_rms", stats.baseline_rms
                      json.field "relative_rms_effect", stats.relative_rms_effect
                      json.field "exact_zero", stats.exact_zero
                      json.field "finite", stats.finite
                    end
                  end
                end
              end
            end
          end
        end
      end
      json.field "interpretation_scope", "conditional numerical refiner factorization; not semantic or perceptual control evidence"
      json.field "passed", !parity_failed && invariants_passed
    end
  end
  write_fresh_output(report, report_json)
  puts report_json
  raise "factorial native/source parity or exact invariants failed; report preserved at #{report}" if parity_failed || !invariants_passed
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

def encode_factor_case(chain : ML::Metal::MPSCommandChain,
                       source : ML::MetalBuffer, qwen : ML::MetalBuffer,
                       c_override : ML::MetalBuffer?,
                       refiner : ML::Video::HyNomasTextRefinerMPSPlan,
                       checkpoints : Hash(String, ML::MetalBuffer),
                       destinations : Hash(String, ML::MetalBuffer),
                       input_copy_plan : HyNomasRefinerTransferProbe::BF16CopyPlan,
                       capture_plan : ML::Video::HyNomasContinuationHistoryKernels::AppendPlan) : Nil
  input_copy_plan.encode(chain.encoder, source, qwen, 1280 * 3584)
  if c_override
    raise ArgumentError.new("c override aliases source Qwen buffer") if c_override.same?(qwen)
    refiner.encode_with_probe_c_override(chain, c_override, capture_plan)
  else
    refiner.encode(chain)
  end
  HyNomasRefinerCFactorialProbe::OUTPUT_CHECKPOINTS.each do |checkpoint|
    count = HyNomasRefinerTransferProbe.tensor_elements(checkpoint)
    capture_plan.encode(chain.encoder, checkpoints[checkpoint], destinations[checkpoint], count, 0)
  end
end

def factorial_effects(outputs : Hash(String, Hash(String, Array(Float32)))) : Hash(String, Hash(String, HyNomasRefinerCFactorialProbe::EffectStats))
  result = {} of String => Hash(String, HyNomasRefinerCFactorialProbe::EffectStats)
  {
    "combined_B_minus_A0"  => {"base" => "A0", "changed" => "B"},
    "text_B_minus_A_at_cA" => {"base" => "A0", "changed" => "X10"},
    "c_B_minus_A_at_textA" => {"base" => "A0", "changed" => "X01"},
  }.each do |effect_name, pair|
    values = {} of String => HyNomasRefinerCFactorialProbe::EffectStats
    HyNomasRefinerCFactorialProbe::OUTPUT_CHECKPOINTS.each do |checkpoint|
      base = outputs[pair["base"]][checkpoint]
      changed = outputs[pair["changed"]][checkpoint]
      values[checkpoint] = HyNomasRefinerCFactorialProbe.effect_stats(base, changed)
      if checkpoint == "block0.output" || checkpoint == "block1.output"
        HyNomasRefinerCFactorialProbe::SPANS.each do |span_name, range|
          values["#{checkpoint}.#{span_name}"] = HyNomasRefinerCFactorialProbe.effect_stats(
            HyNomasRefinerTransferProbe.span(base, range[0], range[1]),
            HyNomasRefinerTransferProbe.span(changed, range[0], range[1]))
        end
      end
    end
    result[effect_name] = values
  end
  interaction = {} of String => HyNomasRefinerCFactorialProbe::EffectStats
  HyNomasRefinerCFactorialProbe::OUTPUT_CHECKPOINTS.each do |checkpoint|
    interaction[checkpoint] = HyNomasRefinerCFactorialProbe.interaction_stats(
      outputs["A0"][checkpoint], outputs["B"][checkpoint],
      outputs["X10"][checkpoint], outputs["X01"][checkpoint])
    if checkpoint == "block0.output" || checkpoint == "block1.output"
      HyNomasRefinerCFactorialProbe::SPANS.each do |span_name, range|
        interaction["#{checkpoint}.#{span_name}"] = HyNomasRefinerCFactorialProbe.interaction_stats(
          HyNomasRefinerTransferProbe.span(outputs["A0"][checkpoint], range[0], range[1]),
          HyNomasRefinerTransferProbe.span(outputs["B"][checkpoint], range[0], range[1]),
          HyNomasRefinerTransferProbe.span(outputs["X10"][checkpoint], range[0], range[1]),
          HyNomasRefinerTransferProbe.span(outputs["X01"][checkpoint], range[0], range[1]))
      end
    end
  end
  result["interaction_B_minus_X10_minus_X01_plus_A0"] = interaction
  result
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
