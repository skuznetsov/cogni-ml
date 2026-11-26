require "option_parser"
require "json"
require "digest/sha256"
require "./support/qwen25_rms_statistics_probe"
require "./support/qwen25_prefix_probe"
require "../src/ml/gguf/qwen25_hy_nomas"

{% unless flag?(:cpu_only) %}
  lib Qwen25RMSStatisticsProbeFFI
    fun create = gs_qwen25_mps_rms_statistics_probe_create(
      device : Void*, batches : Int32, tokens : Int32, dim : Int32,
      epsilon : Float32, input : Void*, upstream_square : Void*,
      power_exponent : Void*, power_exponent_int64 : Void*, outputs : Void**,
      error : UInt8*, error_capacity : LibC::SizeT,
    ) : Void*
  end
{% end %}

class ML::MetalBuffer
  @@qwen25_rms_statistics_probe_allocations = 0_i64

  def self.qwen25_rms_statistics_probe_allocations : Int64
    @@qwen25_rms_statistics_probe_allocations
  end

  private def track_alloc(bytes : Int64) : Nil
    @@qwen25_rms_statistics_probe_allocations += 1
    previous_def(bytes)
  end
end

module Qwen25RMSStatisticsProbeRunner
  extend self

  def read_buffer(buffer : ML::MetalBuffer) : Bytes
    raise ArgumentError.new("invalid RMS probe output buffer") unless buffer.valid?
    bytes = Bytes.new(buffer.size.to_i)
    buffer.read_bytes(bytes.to_unsafe, bytes.size)
    bytes
  end

  def compare(reference : Qwen25RMSStatisticsProbe::ReferenceFixture, suffix : String,
              actual : Bytes, row_width : Int64) : Qwen25RMSStatisticsProbe::FloatComparison
    Qwen25RMSStatisticsProbe.compare_f32(
      "layers.0.post_attention_layernorm.#{suffix}",
      reference.expected(suffix), actual, row_width)
  end
end

stage_reference_path = ""
stage_reference_sha256 = ""
rms_reference_path = ""
rms_reference_sha256 = ""
report_path = ""
max_named_bytes = 0_i64
OptionParser.parse do |parser|
  parser.banner = "Isolated Qwen2.5-VL RMS reduction probe; run under the external memory guard."
  parser.on("--stage-reference PATH", "Authenticated layer-0 S1280 residual stage manifest") { |value| stage_reference_path = value }
  parser.on("--stage-reference-sha256 SHA256", "Trusted d29c... source stage manifest digest") { |value| stage_reference_sha256 = value }
  parser.on("--rms-reference PATH", "Authenticated standalone RMS-only oracle manifest") { |value| rms_reference_path = value }
  parser.on("--rms-reference-sha256 SHA256", "Trusted RMS-only manifest digest") { |value| rms_reference_sha256 = value }
  parser.on("--report PATH", "New native comparison report") { |value| report_path = value }
  parser.on("--max-named-bytes N", "Aggregate named Metal-buffer cap (at most 256 MiB)") { |value| max_named_bytes = value.to_i64 }
end
if stage_reference_path.empty? || stage_reference_sha256.empty? ||
   rms_reference_path.empty? || rms_reference_sha256.empty? ||
   report_path.empty? || max_named_bytes <= 0
  raise ArgumentError.new("both authenticated references, report path and positive named-buffer budget required")
end
unless stage_reference_sha256 == Qwen25RMSStatisticsProbe::TRUSTED_STAGE_SHA256
  raise ArgumentError.new("source stage must use the separately pinned d29c... manifest digest")
end
raise ArgumentError.new("refusing to replace native report") if File.exists?(report_path) || File.symlink?(report_path)
raise ArgumentError.new("report directory must already exist") unless Dir.exists?(File.dirname(File.expand_path(report_path)))
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?

source = Qwen25RMSStatisticsProbe::SourceFixture.new(stage_reference_path, stage_reference_sha256)
reference = Qwen25RMSStatisticsProbe::ReferenceFixture.new(
  rms_reference_path, rms_reference_sha256, source)
named_bytes = Qwen25RMSStatisticsProbe.admit_named_bytes!(
  source.tokens, source.dim, max_named_bytes)
STDERR.puts "admitted #{named_bytes} named bytes for isolated layer-0 post-attention RMS at S=#{source.tokens}, D=#{source.dim}"
upstream_square_raw = reference.expected("square_f32")
expected_square_bytes = source.tokens.to_i64 * source.dim * 4
raise ArgumentError.new("authenticated upstream square has unexpected byte length") unless upstream_square_raw.size == expected_square_bytes
power_exponent_raw = Qwen25RMSStatisticsProbe.exponent_two_f32le
power_exponent_int64_raw = Qwen25RMSStatisticsProbe.exponent_two_i64le

owned = [] of ML::MetalBuffer
plan = Pointer(Void).null
chain : ML::Metal::MPSCommandChain? = nil
native_error = Bytes.new(1024, 0_u8)
output_buffers = [] of ML::MetalBuffer

begin
  input = ML::MetalBuffer.new(source.residual.size.to_i64)
  owned << input
  input.write_bytes(source.residual.to_unsafe, source.residual.size)
  upstream_square = ML::MetalBuffer.new(upstream_square_raw.size.to_i64)
  owned << upstream_square
  upstream_square.write_bytes(upstream_square_raw.to_unsafe, upstream_square_raw.size)
  power_exponent = ML::MetalBuffer.new(power_exponent_raw.size.to_i64)
  owned << power_exponent
  power_exponent.write_bytes(power_exponent_raw.to_unsafe, power_exponent_raw.size)
  power_exponent_int64 = ML::MetalBuffer.new(power_exponent_int64_raw.size.to_i64)
  owned << power_exponent_int64
  power_exponent_int64.write_bytes(power_exponent_int64_raw.to_unsafe, power_exponent_int64_raw.size)

  tokens = source.tokens
  dim = source.dim
  output_sizes = [
    tokens.to_i64 * dim * 4, # BF16 -> F32 input panel
    tokens.to_i64 * dim * 4, # separately materialized square panel
    tokens.to_i64 * 4,       # materialized variance
    tokens.to_i64 * 4,       # materialized variance + epsilon
    tokens.to_i64 * 4,       # direct production helper plus epsilon
    tokens.to_i64 * 4,       # compile optimization level 0 plus epsilon
    tokens.to_i64 * 4,       # upstream-square materialized variance
    tokens.to_i64 * 4,       # upstream-square materialized variance + epsilon
    tokens.to_i64 * dim * 4, # F32-[1]-buffer exponent power panel
    tokens.to_i64 * 4,       # F32-buffer exponent variance
    tokens.to_i64 * 4,       # F32-buffer exponent variance + epsilon
    tokens.to_i64 * dim * 4, # matched constant exponent power panel
    tokens.to_i64 * 4,       # constant exponent variance
    tokens.to_i64 * 4,       # constant exponent variance + epsilon
    tokens.to_i64 * dim * 4, # Int64-[1]-buffer exponent cast to F32 power panel
    tokens.to_i64 * 4,       # Int64-buffer exponent variance
    tokens.to_i64 * 4,       # Int64-buffer exponent variance + epsilon
  ]
  output_sizes.each do |bytes|
    buffer = ML::MetalBuffer.new(bytes)
    output_buffers << buffer
    owned << buffer
  end

  output_handles = output_buffers.map(&.handle)
  {% unless flag?(:cpu_only) %}
    plan = Qwen25RMSStatisticsProbeFFI.create(
      ML::Metal::Device.instance.device_handle, 1, tokens, dim,
      source.epsilon, input.handle, upstream_square.handle,
      power_exponent.handle, power_exponent_int64.handle, output_handles.to_unsafe,
      native_error.to_unsafe, native_error.size)
    raise "Qwen2.5-VL RMS statistics probe preparation failed: #{String.new(native_error.to_unsafe)}" if plan.null?
  {% else %}
    raise "Metal disabled (cpu_only)"
  {% end %}

  chain = ML::Metal::MPSCommandChain.new
  allocations_before = ML::MetalBuffer.qwen25_rms_statistics_probe_allocations
  encode_started = Time.instant
  chain.pause!
  {% unless flag?(:cpu_only) %}
    status = Qwen25MPSFFI.encode(
      plan, chain.handle, native_error.to_unsafe, native_error.size)
    chain.resume!
    raise "Qwen2.5-VL RMS statistics probe encode failed (#{status}): #{String.new(native_error.to_unsafe)}" unless status == 0
  {% else %}
    raise "Metal disabled (cpu_only)"
  {% end %}
  encode_seconds = (Time.instant - encode_started).total_seconds
  encode_allocations = ML::MetalBuffer.qwen25_rms_statistics_probe_allocations - allocations_before
  raise "RMS probe encoding allocated named Metal buffers" unless encode_allocations == 0
  raise "RMS probe encoding introduced a device wait" unless chain.wait_count == 0

  wait_started = Time.instant
  chain.commit_and_wait
  wait_seconds = (Time.instant - wait_started).total_seconds
  raise "RMS probe requires exactly one terminal wait" unless chain.wait_count == 1 && chain.completed?

  # Every output read below follows the sole terminal wait.
  input_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[0])
  square_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[1])
  variance_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[2])
  materialized_plus_epsilon_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[3])
  production_plus_epsilon_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[4])
  optimization0_plus_epsilon_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[5])
  upstream_square_variance_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[6])
  upstream_square_plus_epsilon_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[7])
  fed_exponent_square_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[8])
  fed_exponent_variance_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[9])
  fed_exponent_plus_epsilon_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[10])
  constant_exponent_square_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[11])
  constant_exponent_variance_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[12])
  constant_exponent_plus_epsilon_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[13])
  int64_exponent_square_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[14])
  int64_exponent_variance_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[15])
  int64_exponent_plus_epsilon_f32 = Qwen25RMSStatisticsProbeRunner.read_buffer(output_buffers[16])

  row_width = dim.to_i64
  comparisons = [
    Qwen25RMSStatisticsProbeRunner.compare(reference, "input_f32", input_f32, row_width),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "square_f32", square_f32, row_width),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_f32", variance_f32, 1_i64),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_plus_eps_f32", materialized_plus_epsilon_f32, 1_i64),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_plus_eps_f32", production_plus_epsilon_f32, 1_i64),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_plus_eps_f32", optimization0_plus_epsilon_f32, 1_i64),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_f32", upstream_square_variance_f32, 1_i64),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_plus_eps_f32", upstream_square_plus_epsilon_f32, 1_i64),
    Qwen25RMSStatisticsProbe.compare_f32(
      "f32_fed_exponent_power_panel", reference.expected("square_f32"), fed_exponent_square_f32, row_width),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_f32", fed_exponent_variance_f32, 1_i64),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_plus_eps_f32", fed_exponent_plus_epsilon_f32, 1_i64),
    Qwen25RMSStatisticsProbe.compare_f32(
      "f32_constant_exponent_power_panel", reference.expected("square_f32"), constant_exponent_square_f32, row_width),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_f32", constant_exponent_variance_f32, 1_i64),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_plus_eps_f32", constant_exponent_plus_epsilon_f32, 1_i64),
    Qwen25RMSStatisticsProbe.compare_f32(
      "int64_fed_exponent_power_panel", reference.expected("square_f32"), int64_exponent_square_f32, row_width),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_f32", int64_exponent_variance_f32, 1_i64),
    Qwen25RMSStatisticsProbeRunner.compare(reference, "variance_plus_eps_f32", int64_exponent_plus_epsilon_f32, 1_i64),
  ]
  production_vs_materialized = Qwen25RMSStatisticsProbe.compare_f32(
    "materialized_plus_eps_vs_direct_production_plus_eps",
    production_plus_epsilon_f32, materialized_plus_epsilon_f32, 1_i64)
  optimization0_vs_production = Qwen25RMSStatisticsProbe.compare_f32(
    "optimization_level_0_plus_eps_vs_direct_production_plus_eps",
    production_plus_epsilon_f32, optimization0_plus_epsilon_f32, 1_i64)
  upstream_square_vs_production = Qwen25RMSStatisticsProbe.compare_f32(
    "upstream_square_materialized_plus_eps_vs_direct_production_plus_eps",
    production_plus_epsilon_f32, upstream_square_plus_epsilon_f32, 1_i64)
  same_reduction_square_variance = Qwen25RMSStatisticsProbe.compare_f32(
    "native_square_materialized_variance_vs_upstream_square_materialized_variance",
    variance_f32, upstream_square_variance_f32, 1_i64)
  same_reduction_square_plus_epsilon = Qwen25RMSStatisticsProbe.compare_f32(
    "native_square_materialized_plus_eps_vs_upstream_square_materialized_plus_eps",
    materialized_plus_epsilon_f32, upstream_square_plus_epsilon_f32, 1_i64)
  fed_vs_constant_power_panel = Qwen25RMSStatisticsProbe.compare_f32(
    "f32_fed_vs_constant_exponent_power_panel",
    constant_exponent_square_f32, fed_exponent_square_f32, row_width)
  fed_vs_constant_variance = Qwen25RMSStatisticsProbe.compare_f32(
    "f32_fed_vs_constant_exponent_variance",
    constant_exponent_variance_f32, fed_exponent_variance_f32, 1_i64)
  fed_vs_constant_plus_epsilon = Qwen25RMSStatisticsProbe.compare_f32(
    "f32_fed_vs_constant_exponent_plus_epsilon",
    constant_exponent_plus_epsilon_f32, fed_exponent_plus_epsilon_f32, 1_i64)
  torch_scalar_route_vs_f32_fed_power_panel = Qwen25RMSStatisticsProbe.compare_f32(
    "torch_int64_scalar_route_vs_f32_fed_power_panel",
    int64_exponent_square_f32, fed_exponent_square_f32, row_width)
  torch_scalar_route_vs_f32_fed_variance = Qwen25RMSStatisticsProbe.compare_f32(
    "torch_int64_scalar_route_vs_f32_fed_variance",
    int64_exponent_variance_f32, fed_exponent_variance_f32, 1_i64)
  torch_scalar_route_vs_f32_fed_plus_epsilon = Qwen25RMSStatisticsProbe.compare_f32(
    "torch_int64_scalar_route_vs_f32_fed_plus_epsilon",
    int64_exponent_plus_epsilon_f32, fed_exponent_plus_epsilon_f32, 1_i64)
  finite_gate_passed = comparisons.all? { |comparison| comparison[:finite_failures] == 0 } &&
                       production_vs_materialized[:finite_failures] == 0 &&
                       optimization0_vs_production[:finite_failures] == 0 &&
                       upstream_square_vs_production[:finite_failures] == 0 &&
                       same_reduction_square_variance[:finite_failures] == 0 &&
                       same_reduction_square_plus_epsilon[:finite_failures] == 0 &&
                       fed_vs_constant_power_panel[:finite_failures] == 0 &&
                       fed_vs_constant_variance[:finite_failures] == 0 &&
                       fed_vs_constant_plus_epsilon[:finite_failures] == 0 &&
                       torch_scalar_route_vs_f32_fed_power_panel[:finite_failures] == 0 &&
                       torch_scalar_route_vs_f32_fed_variance[:finite_failures] == 0 &&
                       torch_scalar_route_vs_f32_fed_plus_epsilon[:finite_failures] == 0
  raw_gate_passed = comparisons.all? { |comparison| comparison[:raw_unequal] == 0 } &&
                    production_vs_materialized[:raw_unequal] == 0 &&
                    optimization0_vs_production[:raw_unequal] == 0 &&
                    upstream_square_vs_production[:raw_unequal] == 0 &&
                    same_reduction_square_variance[:raw_unequal] == 0 &&
                    same_reduction_square_plus_epsilon[:raw_unequal] == 0 &&
                    fed_vs_constant_power_panel[:raw_unequal] == 0 &&
                    fed_vs_constant_variance[:raw_unequal] == 0 &&
                    fed_vs_constant_plus_epsilon[:raw_unequal] == 0 &&
                    torch_scalar_route_vs_f32_fed_power_panel[:raw_unequal] == 0 &&
                    torch_scalar_route_vs_f32_fed_variance[:raw_unequal] == 0 &&
                    torch_scalar_route_vs_f32_fed_plus_epsilon[:raw_unequal] == 0

  report = JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", "cogni_ml.qwen25_vl_pretrained_rms_statistics_native_probe"
      json.field "schema_version", 3
      json.field "probe_complete", true
      json.field "source_stage_manifest", source.path
      json.field "source_stage_manifest_sha256", source.manifest_sha256
      json.field "rms_reference_manifest", reference.path
      json.field "rms_reference_manifest_sha256", reference.manifest_sha256
      json.field "source_residual_sha256", source.residual_sha256
      json.field "rms_reference_input_sha256", reference.input_sha256
      json.field "authenticated_upstream_square_sha256", Digest::SHA256.hexdigest(upstream_square_raw)
      json.field "authenticated_upstream_square_uploaded_bytes", upstream_square_raw.size
      exponent_control = Qwen25RMSStatisticsProbe::EXPONENT_POWER_CONTROL
      json.field "f32_power_exponent", exponent_control[:value]
      json.field "f32_power_exponent_dtype", exponent_control[:dtype]
      json.field "f32_power_exponent_shape", exponent_control[:shape]
      json.field "f32_power_exponent_bits", exponent_control[:bits]
      json.field "f32_power_exponent_buffer_sha256", Digest::SHA256.hexdigest(power_exponent_raw)
      json.field "f32_power_exponent_buffer_uploaded_bytes", power_exponent_raw.size
      json.field "torch_scalar_power_exponent_dtype", exponent_control[:torch_source_scalar_dtype]
      json.field "torch_scalar_power_exponent_shape", exponent_control[:torch_source_scalar_shape]
      json.field "torch_scalar_power_exponent_raw_sha256", Digest::SHA256.hexdigest(power_exponent_int64_raw)
      json.field "torch_scalar_power_exponent_uploaded_bytes", power_exponent_int64_raw.size
      json.field "torch_scalar_lowering_source_commit", Qwen25RMSStatisticsProbe::TORCH_SOURCE_COMMIT
      json.field "torch_scalar_lowering_source_note_sha256", Qwen25RMSStatisticsProbe::TORCH_SCALAR_LOWERING_SOURCE_SHA256
      json.field "source_bf16_sidecar_sha256", source.bf16_sidecar_sha256
      json.field "source_f32_sidecar_sha256", source.f32_sidecar_sha256
      json.field "rms_f32_sidecar_sha256", reference.f32_sidecar_sha256
      json.field "rms_bf16_sidecar_sha256", reference.bf16_sidecar_sha256
      json.field "norm_weight_key", source.norm_weight["key"].as_s
      json.field "norm_weight_tensor_sha256", source.norm_weight["tensor_sha256"].as_s
      json.field "layer_index", 0
      json.field "rms_site", "post_attention_layernorm"
      json.field "tokens", tokens
      json.field "dim", dim
      json.field "epsilon", source.epsilon
      json.field "named_bytes", named_bytes
      json.field "max_named_bytes", max_named_bytes
      json.field "named_allocations_during_encode", encode_allocations
      json.field "terminal_wait_count", chain.wait_count
      json.field "diagnostic_readbacks_before_terminal_wait", 0
      json.field "diagnostic_readbacks_after_terminal_wait", 17
      json.field "encode_seconds", encode_seconds
      json.field "terminal_wait_seconds", wait_seconds
      json.field "gpu_elapsed_seconds", chain.gpu_elapsed_seconds
      json.field "command_segments", chain.segment_count
      json.field "tolerance", {atol: Qwen25RMSStatisticsProbe::ATOL, rtol: Qwen25RMSStatisticsProbe::RTOL}
      json.field "probe_step_order", [
        "BF16 residual cast to F32 input panel",
        "square F32 input panel into separately bound F32 square panel",
        "same mean-over-axis +2 graph applied to native square and authenticated upstream square; bind variance and variance+epsilon for both",
        "direct gs_qwen25_rms_statistics_step production helper, nil compilation descriptor",
        "same production helper graph with MPSGraphCompilationDescriptor optimizationLevel=0",
        "same F32 primary input and power graph with prepared F32 [1] exponent buffer, then materialized mean/epsilon",
        "matched F32 [1] exponent constant 2.0 through the same power and materialized mean/epsilon graphs",
        "prepared Int64 [1] exponent buffer followed by graph cast to F32, power, and the same materialized mean/epsilon graph",
      ]
      json.field "materialized_reduction", {
        input_shape: [1, tokens, dim], reduction_axis: [2],
        variance_shape: [1, tokens, 1], explicit_reshape: true,
        square_materialized_in_prior_step: true,
      }
      control = Qwen25RMSStatisticsProbe::OPTIMIZATION0_CONTROL
      json.field "optimization_level_0_control" do
        json.object do
          json.field "compilation_descriptor_type", control[:compilation_descriptor_type]
          json.field "compilation_api", control[:compilation_api]
          json.field "optimization_level", control[:optimization_level]
          json.field "graph_shape", [1, tokens, dim]
          json.field "reduction_axis", control[:reduction_axis]
          json.field "target", control[:target]
          json.field "explicit_reshape", control[:explicit_reshape]
          json.field "encode_route", control[:encode_route]
          json.field "executable_descriptor_type", control[:executable_descriptor_type]
          json.field "executable_wait_until_completed", control[:executable_wait_until_completed]
          json.field "differs_from_torch_graph_execution_descriptor_route", control[:differs_from_torch_graph_execution_descriptor_route]
        end
      end
      json.field "torch_route_scope_note",
        "The level-0 control changes the compile descriptor only. It does not reproduce Torch's MPSGraphExecutionDescriptor graph-encode route; a negative result does not refute execution-path, specialization, or cache differences."
      json.field "comparisons", comparisons
      json.field "materialized_plus_eps_vs_direct_production_plus_eps", production_vs_materialized
      json.field "optimization_level_0_plus_eps_vs_direct_production_plus_eps", optimization0_vs_production
      json.field "upstream_square_materialized_plus_eps_vs_direct_production_plus_eps", upstream_square_vs_production
      json.field "same_reduction_native_square_vs_upstream_square_variance", same_reduction_square_variance
      json.field "same_reduction_native_square_vs_upstream_square_plus_epsilon", same_reduction_square_plus_epsilon
      json.field "upstream_square_control_scope",
        "The authenticated upstream square uses the same native axis +2 materialized mean and epsilon graph as the native square; their direct comparison isolates the square-panel values within this precompiled executable route."
      json.field "exponent_power_controls" do
        json.object do
          json.field "matched_f32_constant_vs_buffer" do
            json.object do
              json.field "dtype", exponent_control[:dtype]
              json.field "shape", exponent_control[:shape]
              json.field "value", exponent_control[:value]
              json.field "f32_bits", exponent_control[:bits]
              json.field "operation", exponent_control[:operation]
              json.field "arms_differ_only_by_secondary_operand_source", true
              json.field "fed_buffer_sha256", Digest::SHA256.hexdigest(power_exponent_raw)
              json.field "matches_torch_post_cast_operand", true
              json.field "scope_note", "This matched arm isolates a buffer-fed F32 [1] operand from a same-value F32 graph constant."
            end
          end
          json.field "torch_scalar_int64_feed_then_f32_cast" do
            json.object do
              torch_control = Qwen25RMSStatisticsProbe::TORCH_INT64_EXPONENT_CONTROL
              json.field "source_scalar_dtype", torch_control[:dtype]
              json.field "source_scalar_shape", torch_control[:shape]
              json.field "raw_int64_value", 2
              json.field "cast_before_power", torch_control[:cast_before_power]
              json.field "uploaded_buffer_sha256", Digest::SHA256.hexdigest(power_exponent_int64_raw)
              json.field "uploaded_bytes", power_exponent_int64_raw.size
              json.field "source_commit", Qwen25RMSStatisticsProbe::TORCH_SOURCE_COMMIT
              json.field "source_note_sha256", Qwen25RMSStatisticsProbe::TORCH_SCALAR_LOWERING_SOURCE_SHA256
              json.field "scope_note", "This arm matches Torch's scalar feed dtype/shape and explicit common-dtype cast before power, within the native precompiled executable route; it does not reproduce Torch's graph-encode route."
            end
          end
        end
      end
      json.field "f32_fed_vs_constant_exponent_power_panel", fed_vs_constant_power_panel
      json.field "f32_fed_vs_constant_exponent_variance", fed_vs_constant_variance
      json.field "f32_fed_vs_constant_exponent_plus_epsilon", fed_vs_constant_plus_epsilon
      json.field "torch_int64_route_vs_f32_fed_power_panel", torch_scalar_route_vs_f32_fed_power_panel
      json.field "torch_int64_route_vs_f32_fed_variance", torch_scalar_route_vs_f32_fed_variance
      json.field "torch_int64_route_vs_f32_fed_plus_epsilon", torch_scalar_route_vs_f32_fed_plus_epsilon
      json.field "f32_fed_vs_constant_exponent_raw_bitwise_parity_passed",
        fed_vs_constant_power_panel[:raw_unequal] == 0 && fed_vs_constant_variance[:raw_unequal] == 0 && fed_vs_constant_plus_epsilon[:raw_unequal] == 0
      json.field "f32_fed_vs_constant_exponent_finite_tolerance_passed",
        fed_vs_constant_power_panel[:finite_failures] == 0 && fed_vs_constant_variance[:finite_failures] == 0 && fed_vs_constant_plus_epsilon[:finite_failures] == 0
      json.field "torch_int64_cast_route_raw_bitwise_parity_passed",
        comparisons[14][:raw_unequal] == 0 && comparisons[15][:raw_unequal] == 0 && comparisons[16][:raw_unequal] == 0
      json.field "torch_int64_cast_route_finite_tolerance_passed",
        comparisons[14][:finite_failures] == 0 && comparisons[15][:finite_failures] == 0 && comparisons[16][:finite_failures] == 0
      json.field "upstream_square_control_raw_bitwise_parity_passed",
        comparisons[6][:raw_unequal] == 0 && comparisons[7][:raw_unequal] == 0
      json.field "upstream_square_control_finite_tolerance_passed",
        comparisons[6][:finite_failures] == 0 && comparisons[7][:finite_failures] == 0
      json.field "finite_tolerance_gate_passed", finite_gate_passed
      json.field "raw_bitwise_parity_passed", raw_gate_passed
      json.field "direct_production_plus_eps_raw_equals_materialized_plus_eps",
        production_vs_materialized[:raw_unequal] == 0
      json.field "optimization_level_0_raw_equals_direct_production",
        optimization0_vs_production[:raw_unequal] == 0
      json.field "upstream_square_materialized_raw_equals_direct_production",
        upstream_square_vs_production[:raw_unequal] == 0
      json.field "full_rms_normalization_or_gain_executed", false
      json.field "interpretation", "This probe reports measured reduction intermediates. A finite-tolerance pass does not establish raw-bit parity or S1280 qualification."
    end
  end
  File.write(report_path, report)
  puts "probe_complete=true finite_tolerance_gate_passed=#{finite_gate_passed} raw_bitwise_parity_passed=#{raw_gate_passed} named_bytes=#{named_bytes} terminal_waits=#{chain.wait_count} encode_allocations=#{encode_allocations}"
  comparisons.each { |comparison| puts comparison.to_json }
  puts production_vs_materialized.to_json
  puts optimization0_vs_production.to_json
  puts upstream_square_vs_production.to_json
  puts same_reduction_square_variance.to_json
  puts same_reduction_square_plus_epsilon.to_json
  puts fed_vs_constant_power_panel.to_json
  puts fed_vs_constant_variance.to_json
  puts fed_vs_constant_plus_epsilon.to_json
  puts torch_scalar_route_vs_f32_fed_power_panel.to_json
  puts torch_scalar_route_vs_f32_fed_variance.to_json
  puts torch_scalar_route_vs_f32_fed_plus_epsilon.to_json
ensure
  Qwen25PrefixProbe.cleanup_after_drain(chain) do
    {% unless flag?(:cpu_only) %}
      Qwen25MPSFFI.destroy(plan) unless plan.null?
    {% end %}
    chain.try(&.release)
    owned.reverse_each { |buffer| buffer.release if buffer.valid? }
  end
end
