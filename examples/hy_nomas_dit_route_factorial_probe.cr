require "json"
require "option_parser"
require "./support/hy_nomas_dit_route_factorial_probe"
require "./support/hy_nomas_dit_route_factorial_gpu"

# A diagnostic forward, not a generation entry point. Source admission finishes
# before Runner can allocate a GPU resource. Saved inputs and masks are fixed;
# no scheduler, decoder, output video, or prompt rewrite is part of this probe.
reference = ""
reference_sha256 = ""
report = ""
max_named_bytes = 20_i64 * 1024 * 1024 * 1024
admit_only = false
plan_only = false
self_test_only = false
OptionParser.parse do |parser|
  parser.banner = "Authenticated Hy-Nomas direct action-text route factorial; no generation."
  parser.on("--reference PATH", "Source factorial manifest") { |value| reference = value }
  parser.on("--reference-sha256 SHA256", "Trusted source manifest digest") { |value| reference_sha256 = value }
  parser.on("--report PATH", "Fresh native manifest beside the source manifest") { |value| report = value }
  parser.on("--max-named-bytes N", "Named GPU allocation ceiling, excluding opaque MPS resources") { |value| max_named_bytes = value.to_i64 }
  parser.on("--admit-only", "Authenticate inputs, masks and source records on CPU") { admit_only = true }
  parser.on("--plan", "Print the pure allocation plan; initialize no Metal resources") { plan_only = true }
  parser.on("--self-test", "Run CPU label and tolerance falsifiers only") { self_test_only = true }
end
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
raise ArgumentError.new("named-buffer budget must be positive") unless max_named_bytes > 0
raise ArgumentError.new("choose only one admission mode") if [admit_only, plan_only, self_test_only].count(true) > 1
HyNomasDiTRouteFactorialProbe.self_test!
if self_test_only
  puts({"self_test" => "passed", "metal_resources_created" => false}.to_json)
  exit
end
if plan_only
  puts HyNomasDiTRouteFactorialGPU.memory_plan.to_json
  exit
end
raise ArgumentError.new("reference and trusted digest are required") if reference.empty? || reference_sha256.empty?
raise ArgumentError.new("report is required for a GPU run") if report.empty? && !admit_only
raise ArgumentError.new("admission does not accept a report") if admit_only && !report.empty?
fixture = HyNomasDiTRouteFactorialProbe::Reference.new(reference, reference_sha256)
if admit_only
  puts({"admitted" => true, "reference_sha256" => fixture.sha256,
        "output_tensors" => fixture.output_tensors.size,
        "metal_resources_created" => false}.to_json)
  exit
end

reference_dir = File.dirname(fixture.path)
unless Path[report].absolute? && File.realpath(File.dirname(report)) == reference_dir
  raise ArgumentError.new("native report must be a fresh absolute path beside its source reference")
end
report = File.join(reference_dir, Path[report].basename)
payload = report + ".f32le"
[report, payload].each do |path|
  raise ArgumentError.new("refusing to replace native output: #{path}") if File.exists?(path) || File.symlink?(path)
end

runner : HyNomasDiTRouteFactorialGPU::Runner? = nil
begin
  runner = HyNomasDiTRouteFactorialGPU::Runner.new(fixture, max_named_bytes)
  result = runner.run
  # All diagnostic host reads and comparisons occur after the sole terminal
  # wait; none is an activation dependency in the inference command chain.
  outputs = result[:outputs]
  parity = {} of String => HyNomasDiTRouteFactorialProbe::Comparison
  aa = {} of String => HyNomasDiTRouteFactorialProbe::Comparison
  passed = true
  outputs.each do |name, values|
    unless values.all? { |value| value.finite? && (value.unsafe_as(UInt32) & 0xffff_u32).zero? }
      raise "native output is not finite lossless BF16 expansion: #{name}"
    end
    comparison = HyNomasDiTRouteFactorialProbe.compare_values(fixture.expected_values(name), values)
    parity[name] = comparison
    passed &&= comparison.failures == 0
  end
  HyNomasDiTRouteFactorialGPU::CHECKPOINTS.each do |checkpoint|
    comparison = HyNomasDiTRouteFactorialProbe.compare_values(outputs["M11.#{checkpoint}"], outputs["M11_repeat.#{checkpoint}"])
    aa[checkpoint] = comparison
    passed &&= comparison.failures == 0 && comparison.exact_bit_differences == 0
  end
  parent_velocity = HyNomasDiTRouteFactorialProbe.compare_values(
    fixture.inputs["t1.step1.velocity"], outputs["M11.velocity"])
  passed &&= parent_velocity.failures == 0
  block0_text = {} of String => HyNomasDiTRouteFactorialProbe::Comparison
  block0_protected_differences = {} of String => Int64
  base_labels = fixture.query_labels_for("M11")
  HyNomasDiTRouteFactorialGPU::CASE_ORDER.each do |name|
    comparison = HyNomasDiTRouteFactorialProbe.compare_values(
      outputs["M11.block0.text_valid"], outputs["#{name}.block0.text_valid"])
    block0_text[name] = comparison
    passed &&= comparison.exact_bit_differences == 0
    labels = fixture.query_labels_for(name)
    permitted = Array(Bool).new(HyNomasDiTRouteFactorialGPU::VIDEO) do |row|
      labels[row] != base_labels[row]
    end
    differences = HyNomasDiTRouteFactorialProbe.exact_protected_row_bit_differences(
      outputs["M11.block0.image"], outputs["#{name}.block0.image"],
      HyNomasDiTRouteFactorialGPU::DIM, permitted)
    block0_protected_differences[name] = differences
    passed &&= differences == 0
  end
  block0_interaction_nonzero = HyNomasDiTRouteFactorialProbe.four_arm_interaction_nonzero_count(
    outputs["M11.block0.image"], outputs["M00.block0.image"],
    outputs["M10.block0.image"], outputs["M01.block0.image"])
  passed &&= block0_interaction_nonzero == 0

  records = {} of String => NamedTuple(shape: Array(Int32), dtype: String, source_dtype: String,
    offset_bytes: Int64, nbytes: Int64, sha256: String)
  cursor = 0_i64
  HyNomasDiTRouteFactorialProbe.with_fresh_output(payload) do |io|
    HyNomasDiTRouteFactorialGPU::CASE_ORDER.each do |case_name|
      HyNomasDiTRouteFactorialGPU::CHECKPOINTS.each do |checkpoint|
        name = "#{case_name}.#{checkpoint}"
        values = outputs[name]
        raw = Slice(UInt8).new(values.to_unsafe.as(UInt8*), values.size * sizeof(Float32))
        # This probe targets little-endian Apple Silicon, like the qualified
        # native/source captures. The encoding is explicit in the manifest.
        io.write(raw)
        records[name] = {shape: HyNomasDiTRouteFactorialProbe::OUTPUT_SHAPES[checkpoint],
                         dtype: "F32", source_dtype: "BF16", offset_bytes: cursor,
                         nbytes: raw.size.to_i64, sha256: Digest::SHA256.hexdigest(raw)}
        cursor += raw.size
      end
    end
  end
  report_json = JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", "hy-nomas-dit-route-factorial-native-v1"
      json.field "schema_version", 1
      json.field "reference", fixture.path
      json.field "reference_sha256", fixture.sha256
      json.field "geometry", fixture.document["geometry"]
      json.field "case_factors", fixture.document["case_factors"]
      json.field "mask_proof", fixture.document["mask_proof"]
      json.field "tolerances", {"atol" => HyNomasDiTRouteFactorialProbe::ATOL,
                                "rtol" => HyNomasDiTRouteFactorialProbe::RTOL}
      json.field "resource_accounting", runner.plan
      json.field "max_named_bytes", max_named_bytes
      json.field "opaque_mps_resources_included", false
      json.field "execution" do
        json.object do
          json.field "case_order", HyNomasDiTRouteFactorialGPU::CASE_ORDER
          json.field "blocks", HyNomasDiTRouteFactorialGPU::BLOCKS
          json.field "schedule", "layer-outer; T0 once then five T1 states; one-layer K/V and shared scratch"
          json.field "command_chains", 1
          json.field "terminal_waits", result[:terminal_waits]
          json.field "named_allocations_during_encode", result[:encode_allocations]
          json.field "activation_readbacks_before_terminal_wait", result[:encode_readbacks]
          json.field "postwait_checkpoint_readbacks", records.size
          json.field "forward_seconds", result[:seconds]
          json.field "scheduler_updates", 0
          json.field "decoder_calls", 0
        end
      end
      json.field "payload", {"basename" => Path[payload].basename, "bytes" => cursor,
                             "sha256" => HyNomasDiTRouteFactorialProbe.sha256_file(payload),
                             "encoding" => "little-endian F32 lossless BF16 expansion"}
      json.field "tensors", records
      json.field "source_parity" do
        json.object { parity.each { |name, value| json.field(name) { write_comparison(json, value) } } }
      end
      json.field "native_AA" do
        json.object { aa.each { |name, value| json.field(name) { write_comparison(json, value) } } }
      end
      json.field("accepted_parent_velocity") { write_comparison(json, parent_velocity) }
      json.field "block0_text_invariant" do
        json.object { block0_text.each { |name, value| json.field(name) { write_comparison(json, value) } } }
      end
      json.field "block0_protected_image_bit_differences", block0_protected_differences
      json.field "block0_image_interaction_nonzero", block0_interaction_nonzero
      json.field "passed", passed
      json.field "interpretation_scope", "Frozen-input numerical route sensitivity; no visual acting or semantic quality conclusion."
    end
  end
  HyNomasDiTRouteFactorialProbe.with_fresh_output(report) { |io| io.puts(report_json) }
  puts({"report" => report, "passed" => passed, "source_checks" => parity.size,
        "max_abs" => parity.values.max_of(&.max_abs),
        "forward_seconds" => result[:seconds],
        "named_bytes" => runner.plan["aggregate_named_bytes"]}.to_json)
  raise "native parity, replay, repeat or exact locality failed; diagnostic payload preserved" unless passed
ensure
  runner.try(&.release)
end

def write_comparison(json : JSON::Builder, value : HyNomasDiTRouteFactorialProbe::Comparison) : Nil
  json.object do
    json.field "elements", value.elements
    json.field "failures", value.failures
    json.field "max_abs", value.max_abs
    json.field "rms_abs", value.rms_abs
    json.field "exact_bit_differences", value.exact_bit_differences
  end
end
