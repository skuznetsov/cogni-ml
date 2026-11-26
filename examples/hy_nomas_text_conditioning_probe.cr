require "option_parser"
require "./support/hy_nomas_text_session"
require "./support/hy_nomas_probe_instrumentation"
require "./support/qwen25_prefix_probe"

reference = ""
reference_sha256 = ""
qwen_reference = ""
qwen_reference_sha256 = ""
report = ""
budget = 0_i64
OptionParser.parse do |parser|
  parser.banner = "Bounded saved-conditioning refiner/block control; no live Qwen model or generation."
  parser.on("--reference PATH", "Fresh authenticated refiner/block fixture") { |v| reference = v }
  parser.on("--reference-sha256 SHA256", "Trusted oracle manifest digest") { |v| reference_sha256 = v }
  parser.on("--qwen-reference PATH", "Authenticated saved Qwen conditioning source") { |v| qwen_reference = v }
  parser.on("--qwen-reference-sha256 SHA256", "Trusted Qwen source digest") { |v| qwen_reference_sha256 = v }
  parser.on("--report PATH", "New post-terminal native comparison report") { |v| report = v }
  parser.on("--max-named-bytes N", "Aggregate named allocation cap") { |v| budget = v.to_i64 }
end
raise ArgumentError.new("reference digests, Qwen source, report and positive budget required") if reference.empty? || reference_sha256.empty? || qwen_reference.empty? || qwen_reference_sha256.empty? || report.empty? || budget <= 0
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
raise ArgumentError.new("refusing to replace native report") if File.exists?(report) || File.symlink?(report)
fixture = HyNomasTextProbe::Fixture.new(reference, reference_sha256, qwen_reference_sha256)
named = HyNomasTextProbe::Session.admit(fixture, budget)
source_bytes = fixture.source_conditioning(qwen_reference)
raise ArgumentError.new("saved conditioning source exceeds aggregate named budget") unless named + source_bytes.size <= budget
source : ML::MetalBuffer? = nil
session : HyNomasTextProbe::Session? = nil
chain : ML::Metal::MPSCommandChain? = nil
begin
  source = ML::MetalBuffer.new(source_bytes.size.to_i64)
  source.write_bytes(source_bytes.to_unsafe, source_bytes.size)
  session = HyNomasTextProbe::Session.new(fixture, source, fixture.mask, named)
  chain = ML::Metal::MPSCommandChain.new
  before_allocations = ML::MetalBuffer.hy_text_probe_allocations
  before_reads = ML::MetalBuffer.hy_text_probe_readbacks
  started = Time.instant
  session.encode(chain)
  encode_seconds = (Time.instant - started).total_seconds
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - before_allocations
  prewait_reads = ML::MetalBuffer.hy_text_probe_readbacks - before_reads
  raise "encoding allocated named GPU buffers" unless encode_allocations == 0
  raise "encoding read back activations" unless prewait_reads == 0
  raise "encoding introduced an extra device wait" unless chain.wait_count == 0
  early_release_rejected = false
  begin
    session.release
  rescue ex : ArgumentError
    raise ex unless ex.message.try(&.includes?("command chain"))
    early_release_rejected = true
  end
  raise "in-flight consumer release was accepted" unless early_release_rejected
  started = Time.instant
  chain.commit_and_wait
  wait_seconds = (Time.instant - started).total_seconds
  raise "expected one terminal wait" unless chain.wait_count == 1
  comparisons = session.compare
  reads = ML::MetalBuffer.hy_text_probe_readbacks - before_reads
  raise "unexpected diagnostic read count" unless reads == comparisons.size
  failures = comparisons.select { |row| row[:failures] > 0 }
  File.write(report, JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", "cogni_ml.hy_nomas_text_conditioning_native_control"
      json.field "schema_version", 1
      json.field "reference", fixture.path
      json.field "reference_sha256", fixture.sha256
      json.field "input_scope", "authenticated saved Qwen state26 crop; live producer is not executed by this control"
      json.field "precision_scope", "BF16 resident weights; F32 math and attention; no CUDA/BF16 activation parity claim"
      json.field "named_bytes", named + source_bytes.size
      json.field "named_allocations_during_encode", encode_allocations
      json.field "activation_readbacks_before_terminal_wait", prewait_reads
      json.field "terminal_wait_count", chain.wait_count
      json.field "diagnostic_readbacks_after_terminal_wait", reads
      json.field "early_release_rejected", early_release_rejected
      json.field "encode_seconds", encode_seconds
      json.field "terminal_wait_seconds", wait_seconds
      json.field "gpu_elapsed_seconds", chain.gpu_elapsed_seconds
      json.field "tolerance", {atol: HyNomasTextProbe::ATOL, rtol: HyNomasTextProbe::RTOL}
      json.field "comparisons", comparisons
      json.field "first_mismatch", failures.first?.try(&.[:name])
      json.field "passed", failures.empty?
    end
  end)
  comparisons.each { |row| puts row.to_json }
  raise "first native refiner/block mismatch: #{failures.first[:name]}" unless failures.empty?
  puts "PASS: saved BF16 conditioning -> full native refiner -> actual block0; one terminal wait"
ensure
  Qwen25PrefixProbe.cleanup_after_drain(chain) do
    session.try(&.release)
    chain.try(&.release)
    source.try(&.release)
  end
end
