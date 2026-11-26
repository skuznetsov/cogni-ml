require "option_parser"
require "json"
require "./support/qwen25_connected_rms_probe"
require "./support/qwen25_prefix_probe"
require "../src/ml/gguf/qwen25_hy_nomas"

{% unless flag?(:cpu_only) %}
  lib Qwen25ConnectedRMSProbeFFI
    fun create = gs_qwen25_mps_connected_rms_probe_create(
      device : Void*, batches : Int32, tokens : Int32, dim : Int32,
      epsilon : Float32, input : Void*, gain : Void*, exponent : Void*,
      outputs : Void**, error : UInt8*, error_capacity : LibC::SizeT,
    ) : Void*
  end
{% end %}

class ML::MetalBuffer
  @@qwen25_connected_rms_probe_allocations = 0_i64

  def self.qwen25_connected_rms_probe_allocations : Int64
    @@qwen25_connected_rms_probe_allocations
  end

  private def track_alloc(bytes : Int64) : Nil
    @@qwen25_connected_rms_probe_allocations += 1
    previous_def(bytes)
  end
end

stage_path = ""
stage_sha = ""
rms_path = ""
rms_sha = ""
model_dir = ""
report_path = ""
budget = 0_i64
OptionParser.parse do |parser|
  parser.banner = "Bounded connected RMS/gain diagnostic; run under the external memory guard."
  parser.on("--stage-reference PATH", "Authenticated layer-0 S1280 residual manifest") { |value| stage_path = value }
  parser.on("--stage-reference-sha256 SHA256", "Trusted source manifest digest") { |value| stage_sha = value }
  parser.on("--rms-reference PATH", "Authenticated standalone upstream RMS manifest") { |value| rms_path = value }
  parser.on("--rms-reference-sha256 SHA256", "Trusted upstream RMS manifest digest") { |value| rms_sha = value }
  parser.on("--model-dir PATH", "Checkpoint directory; only the selected BF16 norm gain is read") { |value| model_dir = value }
  parser.on("--report PATH", "New native comparison report") { |value| report_path = value }
  parser.on("--max-named-bytes N", "Named Metal buffer budget, at most 256 MiB") { |value| budget = value.to_i64 }
end
if {stage_path, stage_sha, rms_path, rms_sha, model_dir, report_path}.any?(&.empty?) || budget <= 0 || !ARGV.empty?
  raise ArgumentError.new("authenticated references, model directory, new report and positive budget required")
end
unless stage_sha == Qwen25RMSStatisticsProbe::TRUSTED_STAGE_SHA256
  raise ArgumentError.new("source stage must use the separately pinned d29c... manifest digest")
end
raise ArgumentError.new("refusing to replace native report") if File.exists?(report_path) || File.symlink?(report_path)
raise ArgumentError.new("report directory must exist") unless Dir.exists?(File.dirname(File.expand_path(report_path)))

source = Qwen25RMSStatisticsProbe::SourceFixture.new(stage_path, stage_sha)
reference = Qwen25RMSStatisticsProbe::ReferenceFixture.new(rms_path, rms_sha, source)
named_bytes = Qwen25ConnectedRMSProbe.admit_named_bytes!(source.tokens, source.dim, budget)
layout = Qwen25ConnectedRMSProbe.output_layout(source.tokens, source.dim)
gain_source = Qwen25ConnectedRMSProbe.load_gain(model_dir, source.norm_weight, source.dim)
exponent_raw = Qwen25RMSStatisticsProbe.exponent_two_f32le
expected = layout.map { |output| reference.expected(output[:suffix]) }
STDERR.puts "admitted #{named_bytes} named bytes for connected layer-0 post-attention RMS at S=#{source.tokens}, D=#{source.dim}"

owned = [] of ML::MetalBuffer
outputs = [] of ML::MetalBuffer
plan = Pointer(Void).null
chain : ML::Metal::MPSCommandChain? = nil
error = Bytes.new(1024, 0_u8)
begin
  {source.residual, gain_source[:raw], exponent_raw}.each do |raw|
    buffer = ML::MetalBuffer.new(raw.size.to_i64)
    owned << buffer
    buffer.write_bytes(raw.to_unsafe, raw.size)
  end
  layout.each do |output|
    buffer = ML::MetalBuffer.new(output[:nbytes])
    owned << buffer
    outputs << buffer
  end
  raise "connected RMS named accounting disagrees with allocated buffers" unless owned.sum(&.size) == named_bytes
  output_handles = outputs.map(&.handle)
  {% unless flag?(:cpu_only) %}
    plan = Qwen25ConnectedRMSProbeFFI.create(
      ML::Metal::Device.instance.device_handle, 1, source.tokens, source.dim,
      source.epsilon, owned[0].handle, owned[1].handle, owned[2].handle,
      output_handles.to_unsafe, error.to_unsafe, error.size)
    raise "connected RMS preparation failed: #{String.new(error.to_unsafe)}" if plan.null?
  {% else %}
    raise "Metal disabled (cpu_only)"
  {% end %}

  chain = ML::Metal::MPSCommandChain.new
  allocations_before = ML::MetalBuffer.qwen25_connected_rms_probe_allocations
  started = Time.instant
  chain.pause!
  {% unless flag?(:cpu_only) %}
    status = Qwen25MPSFFI.encode(plan, chain.handle, error.to_unsafe, error.size)
    chain.resume!
    raise "connected RMS encode failed (#{status}): #{String.new(error.to_unsafe)}" unless status == 0
  {% else %}
    raise "Metal disabled (cpu_only)"
  {% end %}
  encode_seconds = (Time.instant - started).total_seconds
  encode_allocations = ML::MetalBuffer.qwen25_connected_rms_probe_allocations - allocations_before
  raise "connected RMS encoded new named buffers" unless encode_allocations == 0
  raise "connected RMS added an intermediate wait" unless chain.wait_count == 0
  started = Time.instant
  chain.commit_and_wait
  wait_seconds = (Time.instant - started).total_seconds
  raise "connected RMS requires one terminal wait" unless chain.wait_count == 1 && chain.completed?

  # Only diagnostic comparisons/readbacks follow the one terminal drain.
  actual = outputs.map do |buffer|
    bytes = Bytes.new(buffer.size.to_i)
    buffer.read_bytes(bytes.to_unsafe, bytes.size)
    bytes
  end
  comparisons = layout.map_with_index do |output, index|
    Qwen25ConnectedRMSProbe.compare(output, expected[index], actual[index])
  end
  fed_vs_legacy = (0...5).map do |index|
    Qwen25ConnectedRMSProbe.compare(layout[index + 5], actual[index], actual[index + 5])
  end
  production_vs_trace = [6, 8, 9].map_with_index do |trace_index, offset|
    Qwen25ConnectedRMSProbe.compare(layout[10 + offset], actual[trace_index], actual[10 + offset])
  end
  candidate_comparisons = comparisons[5, 8] + production_vs_trace
  candidate_finite = candidate_comparisons.all? { |comparison| comparison[:finite_failures] == 0 }
  candidate_raw = candidate_comparisons.all? { |comparison| comparison[:raw_unequal] == 0 }

  report = JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", "cogni_ml.qwen25_vl_pretrained_connected_rms_native_probe"
      json.field "schema_version", 1
      json.field "probe_complete", true
      json.field "source_stage_manifest", source.path
      json.field "source_stage_manifest_sha256", source.manifest_sha256
      json.field "rms_reference_manifest", reference.path
      json.field "rms_reference_manifest_sha256", reference.manifest_sha256
      json.field "source_residual_sha256", source.residual_sha256
      json.field "rms_reference_input_sha256", reference.input_sha256
      json.field "source_bf16_sidecar_sha256", source.bf16_sidecar_sha256
      json.field "source_f32_sidecar_sha256", source.f32_sidecar_sha256
      json.field "rms_bf16_sidecar_sha256", reference.bf16_sidecar_sha256
      json.field "rms_f32_sidecar_sha256", reference.f32_sidecar_sha256
      json.field "gain" do
        json.object do
          json.field "key", source.norm_weight["key"].as_s
          json.field "tensor_sha256", gain_source[:tensor_sha256]
          json.field "uploaded_bytes", gain_source[:raw].size
          json.field "shard_path", gain_source[:path]
          json.field "shard_nbytes", gain_source[:shard_nbytes]
          json.field "header_sha256", gain_source[:header_sha256]
          json.field "full_shard_read_or_upload", false
        end
      end
      json.field "exponent", Qwen25RMSStatisticsProbe::EXPONENT_POWER_CONTROL
      json.field "exponent_buffer_sha256", Digest::SHA256.hexdigest(exponent_raw)
      json.field "tokens", source.tokens
      json.field "dim", source.dim
      json.field "epsilon", source.epsilon
      json.field "named_bytes", named_bytes
      json.field "max_named_bytes", budget
      json.field "named_allocations_during_encode", encode_allocations
      json.field "terminal_wait_count", chain.wait_count
      json.field "diagnostic_readbacks_before_terminal_wait", 0
      json.field "diagnostic_readbacks_after_terminal_wait", outputs.size
      json.field "encode_seconds", encode_seconds
      json.field "terminal_wait_seconds", wait_seconds
      json.field "gpu_elapsed_seconds", chain.gpu_elapsed_seconds
      json.field "command_segments", chain.segment_count
      json.field "prepared_step_count", 12
      json.field "statistics_graph_targets", ["variance_plus_epsilon only"]
      json.field "statistics_graph_reduction_axis", [2]
      json.field "tolerance", {atol: Qwen25RMSStatisticsProbe::ATOL, rtol: Qwen25RMSStatisticsProbe::RTOL}
      json.field "output_layout", layout
      json.field "comparisons", comparisons
      json.field "fed_trace_vs_legacy_trace", fed_vs_legacy
      json.field "fed_production_vs_fed_trace", production_vs_trace
      json.field "arms" do
        json.object do
          {"legacy_trace", "fed_trace", "fed_production"}.each do |arm|
            subset = comparisons.select { |comparison| comparison[:name].starts_with?("#{arm}.") }
            json.field arm do
              json.object do
                json.field "finite_tolerance_passed", subset.all? { |comparison| comparison[:finite_failures] == 0 }
                json.field "raw_bitwise_parity_passed", subset.all? { |comparison| comparison[:raw_unequal] == 0 }
                json.field "negative_control", arm == "legacy_trace"
              end
            end
          end
        end
      end
      json.field "candidate_finite_tolerance_passed", candidate_finite
      json.field "candidate_raw_bitwise_parity_passed", candidate_raw
      json.field "production_default_changed", false
      json.field "full_block_or_prefix_executed", false
      json.field "interpretation", "Fixed-input connected RMS only. The production-style arm overwrites its epsilon scratch with inverse-root; it does not independently retain epsilon or normalized F32. Legacy is a separate negative control. No performance or full-prefix qualification follows."
    end
  end
  File.write(report_path, report)
  puts "probe_complete=true candidate_finite=#{candidate_finite} candidate_raw=#{candidate_raw} named_bytes=#{named_bytes} terminal_waits=#{chain.wait_count} encode_allocations=#{encode_allocations}"
  comparisons.each { |comparison| puts comparison.to_json }
ensure
  Qwen25PrefixProbe.cleanup_after_drain(chain) do
    {% unless flag?(:cpu_only) %}
      Qwen25MPSFFI.destroy(plan) unless plan.null?
    {% end %}
    chain.try(&.release)
    owned.reverse_each { |buffer| buffer.release if buffer.valid? }
  end
end
