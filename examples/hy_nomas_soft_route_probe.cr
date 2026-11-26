require "json"
require "option_parser"
require "./support/hy_nomas_soft_route_probe"
require "./support/hy_nomas_dit_route_factorial_gpu"

# Frozen-input forward only: no scheduler, decoder or video generation.
reference = ""
reference_sha256 = ""
bias_manifest = ""
bias_sha256 = ""
report = ""
budget = 21_i64 * 1024 * 1024 * 1024
plan_only = false
admit_only = false
OptionParser.parse do |parser|
  parser.on("--reference PATH", "Accepted direct-route parent") { |value| reference = value }
  parser.on("--reference-sha256 SHA", "Trusted parent digest") { |value| reference_sha256 = value }
  parser.on("--bias-manifest PATH", "Pinned authored priors") { |value| bias_manifest = value }
  parser.on("--bias-sha256 SHA", "Trusted bias manifest digest") { |value| bias_sha256 = value }
  parser.on("--report PATH", "Fresh native output beside bias manifest") { |value| report = value }
  parser.on("--max-named-bytes N", "Named allocation ceiling") { |value| budget = value.to_i64 }
  parser.on("--plan", "Pure allocation plan; no GPU resources") { plan_only = true }
  parser.on("--admit-only", "Authenticate metadata and parent; no GPU resources") { admit_only = true }
end
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
raise ArgumentError.new("choose one admission mode") if plan_only && admit_only
raise ArgumentError.new("positive named allocation ceiling required") unless budget > 0
bias = HyNomasSoftRouteProbe::Biases.new(bias_manifest, bias_sha256)
if plan_only
  puts HyNomasDiTRouteFactorialGPU.memory_plan(bias.values).to_json
  exit
end
fixture = HyNomasDiTRouteFactorialProbe::Reference.new(reference, reference_sha256)
if admit_only
  puts({"admitted" => true, "metal_resources_created" => false,
        "parent_sha256" => fixture.sha256, "bias_sha256" => bias.sha256}.to_json)
  exit
end
unless Path[report].absolute? && File.realpath(File.dirname(report)) == File.dirname(bias.path)
  raise ArgumentError.new("native report must be beside the pinned bias manifest")
end
payload = report + ".f32le"
[report, payload].each do |path|
  raise ArgumentError.new("refusing to replace native output") if File.exists?(path) || File.symlink?(path)
end
runner : HyNomasDiTRouteFactorialGPU::Runner? = nil
begin
  runner = HyNomasDiTRouteFactorialGPU::Runner.new(fixture, budget, bias.values)
  result = runner.run
  outputs = result[:outputs]
  checks = {} of String => HyNomasDiTRouteFactorialProbe::Comparison
  HyNomasDiTRouteFactorialGPU::CHECKPOINTS.each do |checkpoint|
    checks["parent.#{checkpoint}"] = HyNomasDiTRouteFactorialProbe.compare_values(
      fixture.expected_values("M11.#{checkpoint}"), outputs["zero.#{checkpoint}"])
    checks["AA.#{checkpoint}"] = HyNomasDiTRouteFactorialProbe.compare_values(
      outputs["zero.#{checkpoint}"], outputs["zero_repeat.#{checkpoint}"])
  end
  passed = checks.values.all? { |value| value.failures == 0 && value.exact_bit_differences == 0 }
  outputs.each do |name, values|
    unless values.all? { |value| value.finite? && (value.unsafe_as(UInt32) & 0xffff_u32).zero? }
      raise "soft-route output is not a finite BF16 expansion: #{name}"
    end
  end
  records = {} of String => NamedTuple(shape: Array(Int32), dtype: String,
    source_dtype: String, offset_bytes: Int64, nbytes: Int64, sha256: String)
  cursor = 0_i64
  HyNomasDiTRouteFactorialProbe.with_fresh_output(payload) do |io|
    HyNomasSoftRouteProbe::CASES.each do |name|
      HyNomasDiTRouteFactorialGPU::CHECKPOINTS.each do |checkpoint|
        key = "#{name}.#{checkpoint}"
        values = outputs[key]
        raw = Slice(UInt8).new(values.to_unsafe.as(UInt8*), values.size * sizeof(Float32))
        io.write(raw)
        records[key] = {shape: HyNomasDiTRouteFactorialProbe::OUTPUT_SHAPES[checkpoint],
                        dtype: "F32", source_dtype: "BF16", offset_bytes: cursor,
                        nbytes: raw.size.to_i64, sha256: Digest::SHA256.hexdigest(raw)}
        cursor += raw.size
      end
    end
  end
  document = {
    "schema" => "hy-nomas-soft-route-native-v1",
    "parent_reference" => fixture.path, "parent_sha256" => fixture.sha256,
    "bias_manifest" => bias.path, "bias_sha256" => bias.sha256,
    "case_order" => HyNomasSoftRouteProbe::CASES,
    "geometry" => fixture.document["geometry"],
    "tolerances" => {"atol" => HyNomasDiTRouteFactorialProbe::ATOL, "rtol" => HyNomasDiTRouteFactorialProbe::RTOL},
    "resource_accounting" => runner.plan,
    "execution" => {"blocks" => HyNomasDiTRouteFactorialGPU::BLOCKS,
                    "command_chains" => 1, "terminal_waits" => result[:terminal_waits],
                    "named_allocations_during_encode" => result[:encode_allocations],
                    "activation_readbacks_before_terminal_wait" => result[:encode_readbacks],
                    "postwait_checkpoint_readbacks" => records.size,
                    "forward_seconds" => result[:seconds], "scheduler_updates" => 0, "decoder_calls" => 0},
    "payload" => {"basename" => Path[payload].basename, "bytes" => cursor,
                  "sha256" => HyNomasDiTRouteFactorialProbe.sha256_file(payload),
                  "encoding" => "little-endian F32 lossless BF16 expansion"},
    "tensors" => records,
    "exact_parent_and_AA" => checks.transform_values { |v| {"failures" => v.failures, "exact_bit_differences" => v.exact_bit_differences, "max_abs" => v.max_abs} },
    "passed_native_invariants" => passed,
    "interpretation_scope" => "Frozen-input numerical enrichment only; source parity and visual acting require separate checks.",
  }
  HyNomasDiTRouteFactorialProbe.with_fresh_output(report) { |io| io.puts(document.to_json) }
  puts({"report" => report, "passed_native_invariants" => passed,
        "forward_seconds" => result[:seconds], "named_bytes" => runner.plan["aggregate_named_bytes"]}.to_json)
  raise "soft-route parent or A/A replay failed; diagnostics preserved" unless passed
ensure
  runner.try(&.release)
end
