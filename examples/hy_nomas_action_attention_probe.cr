require "digest"
require "json"
require "option_parser"
require "../src/ml/video/hy_nomas_history_attention"
require "./support/hy_nomas_action_attention_probe"
require "./support/hy_nomas_history_attention_probe_guard"
require "./support/hy_nomas_probe_instrumentation"

fixture_path = ""
fixture_sha256 = ""
oracle_source = ""
upstream_source = ""
report_path = ""
run_native = false
validate_fixture_only = false
OptionParser.parse do |parser|
  parser.banner = "Opt-in synthetic F32 action-mask probe; no model weights or generation."
  parser.on("--fixture-manifest PATH", "CPU oracle manifest") { |value| fixture_path = value }
  parser.on("--fixture-sha256 SHA256", "Trusted CPU oracle manifest digest") { |value| fixture_sha256 = value }
  parser.on("--oracle-source PATH", "CPU oracle Python source authenticated by the manifest") { |value| oracle_source = value }
  parser.on("--upstream-source PATH", "Pinned upstream nomas_cpt.py source") { |value| upstream_source = value }
  parser.on("--report PATH", "New post-terminal JSON report") { |value| report_path = value }
  parser.on("--run-native", "Explicitly allow the scheduled native Metal probe") { run_native = true }
  parser.on("--validate-fixture-only", "Validate fixture without creating a Metal device") { validate_fixture_only = true }
end
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
raise ArgumentError.new("choose either --validate-fixture-only or --run-native") if validate_fixture_only && run_native
if validate_fixture_only
  raise ArgumentError.new("fixture, digest, oracle source and upstream source are required") if fixture_path.empty? || fixture_sha256.empty? || oracle_source.empty? || upstream_source.empty?
  HyNomasActionAttentionProbe::Fixture.new(fixture_path, fixture_sha256, oracle_source, upstream_source)
  puts "PASS: authenticated CPU action-mask fixture validated; no Metal device created"
  exit(0)
end
raise ArgumentError.new("--run-native is required; the probe is intentionally opt-in") unless run_native
raise ArgumentError.new("fixture, digest, oracle source and upstream source are required") if fixture_path.empty? || fixture_sha256.empty? || oracle_source.empty? || upstream_source.empty?
raise ArgumentError.new("native report path is required") if report_path.empty?
raise ArgumentError.new("refusing to replace native report") if File.exists?(report_path) || File.symlink?(report_path)

fixture = HyNomasActionAttentionProbe::Fixture.new(fixture_path, fixture_sha256, oracle_source, upstream_source)
query_buffer : ML::MetalBuffer? = nil
masked_current_key_buffer : ML::MetalBuffer? = nil
masked_current_value_buffer : ML::MetalBuffer? = nil
legacy_current_key_buffer : ML::MetalBuffer? = nil
legacy_current_value_buffer : ML::MetalBuffer? = nil
history_key_buffer : ML::MetalBuffer? = nil
history_value_buffer : ML::MetalBuffer? = nil
masked_output : ML::MetalBuffer? = nil
legacy_output : ML::MetalBuffer? = nil
masked_workspace : ML::Video::HyNomasHistoryAttentionWorkspace? = nil
legacy_workspace : ML::Video::HyNomasHistoryAttentionWorkspace? = nil
masked_plan : ML::Video::HyNomasHistoryAttentionMPSPlan? = nil
legacy_plan : ML::Video::HyNomasHistoryAttentionMPSPlan? = nil
chain : ML::Metal::MPSCommandChain? = nil

initial_named_bytes = ML::MetalBuffer.stats[:live_bytes]
comparisons = [] of HyNomasActionAttentionProbe::Comparison
encode_allocations = 0_i64
preterminal_readbacks = 0_i64
preterminal_wait_count = 0_i32
terminal_wait_count = 0_i32
diagnostic_readbacks = 0_i64
setup_named_bytes = 0_i64
early_masked_plan_release_rejected = false
early_legacy_plan_release_rejected = false
early_workspace_release_rejected = false
encode_seconds = 0.0
wait_seconds = 0.0

begin
  query_buffer = HyNomasActionAttentionProbe.upload_f32(fixture.query)
  masked_current_key_buffer = HyNomasActionAttentionProbe.upload_f32(fixture.current_key)
  masked_current_value_buffer = HyNomasActionAttentionProbe.upload_f32(fixture.current_value)
  legacy_current_key_buffer = HyNomasActionAttentionProbe.upload_f32(fixture.legacy_current_key)
  legacy_current_value_buffer = HyNomasActionAttentionProbe.upload_f32(fixture.legacy_current_value)
  history_key_buffer = HyNomasActionAttentionProbe.upload_f32(fixture.history_key)
  history_value_buffer = HyNomasActionAttentionProbe.upload_f32(fixture.history_value)
  masked_output = HyNomasActionAttentionProbe.output_buffer
  legacy_output = HyNomasActionAttentionProbe.output_buffer

  masked_geometry = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(
    HyNomasActionAttentionProbe::HISTORY_ROWS,
    HyNomasActionAttentionProbe::QUERY_ROWS,
    HyNomasActionAttentionProbe::HEADS,
    HyNomasActionAttentionProbe::HEAD_DIM,
    HyNomasActionAttentionProbe::MASKED_CURRENT_ROWS,
    query_labels: fixture.query_labels,
    key_labels: fixture.key_labels,
  )
  masked_workspace = ML::Video::HyNomasHistoryAttentionWorkspace.new(
    HyNomasActionAttentionProbe::HISTORY_ROWS,
    HyNomasActionAttentionProbe::QUERY_ROWS,
    HyNomasActionAttentionProbe::HEADS,
    HyNomasActionAttentionProbe::HEAD_DIM,
    HyNomasActionAttentionProbe::MASKED_CURRENT_ROWS,
    query_labels: fixture.query_labels,
    key_labels: fixture.key_labels,
  )
  masked_history_slot = masked_workspace.not_nil!.register_history_pair!(
    history_key_buffer.not_nil!, history_value_buffer.not_nil!
  )
  masked_workspace.not_nil!.freeze_history_inventory!

  # The source legacy path compacts away padded text Q/K rows. Keep its Q buffer
  # at the padded row count, feed only visible current K/V rows, and compare
  # only active output rows. This is an unlabelled legacy control.
  legacy_geometry = ML::Video::HyNomasHistoryAttentionWorkspace.geometry(
    HyNomasActionAttentionProbe::HISTORY_ROWS,
    HyNomasActionAttentionProbe::QUERY_ROWS,
    HyNomasActionAttentionProbe::HEADS,
    HyNomasActionAttentionProbe::HEAD_DIM,
    HyNomasActionAttentionProbe::LEGACY_CURRENT_ROWS,
  )
  legacy_workspace = ML::Video::HyNomasHistoryAttentionWorkspace.new(
    HyNomasActionAttentionProbe::HISTORY_ROWS,
    HyNomasActionAttentionProbe::QUERY_ROWS,
    HyNomasActionAttentionProbe::HEADS,
    HyNomasActionAttentionProbe::HEAD_DIM,
    HyNomasActionAttentionProbe::LEGACY_CURRENT_ROWS,
  )
  legacy_history_slot = legacy_workspace.not_nil!.register_history_pair!(
    history_key_buffer.not_nil!, history_value_buffer.not_nil!
  )
  legacy_workspace.not_nil!.freeze_history_inventory!

  masked_plan = ML::Video::HyNomasHistoryAttentionMPSPlan.new(
    masked_workspace.not_nil!, query_buffer.not_nil!,
    masked_current_key_buffer.not_nil!, masked_current_value_buffer.not_nil!, masked_output.not_nil!
  )
  legacy_plan = ML::Video::HyNomasHistoryAttentionMPSPlan.new(
    legacy_workspace.not_nil!, query_buffer.not_nil!,
    legacy_current_key_buffer.not_nil!, legacy_current_value_buffer.not_nil!, legacy_output.not_nil!
  )
  setup_named_bytes = ML::MetalBuffer.stats[:live_bytes] - initial_named_bytes
  chain = ML::Metal::MPSCommandChain.new

  before_allocations = ML::MetalBuffer.hy_text_probe_allocations
  before_readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  started = Time.instant
  masked_plan.not_nil!.encode(chain.not_nil!, masked_history_slot)
  legacy_plan.not_nil!.encode(chain.not_nil!, legacy_history_slot)
  encode_seconds = (Time.instant - started).total_seconds
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - before_allocations
  preterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - before_readbacks
  preterminal_wait_count = chain.not_nil!.wait_count

  begin
    masked_plan.not_nil!.release
  rescue ex : ArgumentError
    early_masked_plan_release_rejected = HyNomasHistoryAttentionProbe.inflight_chain_release_rejected?(
      chain.not_nil!.completed?, ex.message)
  end
  begin
    legacy_plan.not_nil!.release
  rescue ex : ArgumentError
    early_legacy_plan_release_rejected = HyNomasHistoryAttentionProbe.inflight_chain_release_rejected?(
      chain.not_nil!.completed?, ex.message)
  end
  begin
    masked_workspace.not_nil!.release
  rescue ex : ArgumentError
    early_workspace_release_rejected = ex.message.try(&.includes?("Release all")) || false
  end
  raise "in-flight attention plan release was not rejected" unless early_masked_plan_release_rejected && early_legacy_plan_release_rejected
  raise "in-flight attention workspace release was not rejected" unless early_workspace_release_rejected

  started = Time.instant
  chain.not_nil!.commit_and_wait
  wait_seconds = (Time.instant - started).total_seconds
  terminal_wait_count = chain.not_nil!.wait_count
  raise "expected one terminal wait" unless terminal_wait_count == 1

  actual_masked = masked_output.not_nil!.read(HyNomasActionAttentionProbe::OUTPUT_ELEMENTS)
  actual_legacy = legacy_output.not_nil!.read(HyNomasActionAttentionProbe::OUTPUT_ELEMENTS)
  diagnostic_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - before_readbacks
  actual_legacy_active = Array(Float32).new(HyNomasActionAttentionProbe::ACTIVE_OUTPUT_ELEMENTS) { |index| actual_legacy[index] }
  expected_legacy_active = Array(Float32).new(HyNomasActionAttentionProbe::ACTIVE_OUTPUT_ELEMENTS) { |index| fixture.expected_legacy[index] }
  comparisons = [
    HyNomasActionAttentionProbe.compare("native source-range mask vs pinned CPU F32 oracle", actual_masked, fixture.expected_masked),
    HyNomasActionAttentionProbe.compare("unlabelled legacy control vs pinned CPU F32 oracle", actual_legacy_active, expected_legacy_active),
  ]

  active_delta = 0.0_f64
  HyNomasActionAttentionProbe::ACTIVE_OUTPUT_ELEMENTS.times do |index|
    active_delta = Math.max(active_delta, (actual_masked[index].to_f64 - actual_legacy[index].to_f64).abs)
  end
  mask_changes_output = active_delta > 1.0
  failed = comparisons.any? { |row| row.failures > 0 || !row.finite } || !mask_changes_output
  constraints_passed = encode_allocations == 0 && preterminal_readbacks == 0 &&
                       preterminal_wait_count == 0 && terminal_wait_count == 1 && diagnostic_readbacks == 2
  File.open(report_path, "w") do |file|
    JSON.build(file, indent: "  ") do |json|
      json.object do
        json.field "schema", "cogni_ml.hy_nomas_action_attention_native_probe"
        json.field "schema_version", 1
        json.field "fixture_manifest", File.expand_path(fixture_path)
        json.field "fixture_manifest_sha256", fixture.manifest_sha256
        json.field "upstream_revision", HyNomasActionAttentionProbe::UPSTREAM_REVISION
        json.field "upstream_source_sha256", HyNomasActionAttentionProbe::UPSTREAM_SOURCE_SHA256
        json.field "oracle_source_sha256", Digest::SHA256.hexdigest(File.read(oracle_source))
        json.field "input_scope", "tiny synthetic F32 Q/K/V with visible cached history, two action labels, global scene/head tokens, a gap, and hidden text padding; no model weights or generation"
        json.field "mask_scope", "upstream source range rectangles expanded to Q-by-K; zero-label gaps see global keys only; positive action queries see global plus their own action keys; padded queries are zero"
        json.field "legacy_scope", "unlabelled full attention over compact visible current rows and history; compare active rows only"
        json.field "geometry", {
          history_rows:           HyNomasActionAttentionProbe::HISTORY_ROWS,
          query_rows:             HyNomasActionAttentionProbe::QUERY_ROWS,
          masked_current_rows:    HyNomasActionAttentionProbe::MASKED_CURRENT_ROWS,
          legacy_current_rows:    HyNomasActionAttentionProbe::LEGACY_CURRENT_ROWS,
          heads:                  HyNomasActionAttentionProbe::HEADS,
          head_dim:               HyNomasActionAttentionProbe::HEAD_DIM,
          source_range_count:     fixture.range_count,
          masked_graph_calls:     masked_geometry.graph_calls,
          legacy_graph_calls:     legacy_geometry.graph_calls,
          masked_workspace_bytes: masked_geometry.workspace_bytes,
          legacy_workspace_bytes: legacy_geometry.workspace_bytes,
          masked_max_score_bytes: masked_geometry.max_score_bytes,
          legacy_max_score_bytes: legacy_geometry.max_score_bytes,
        }
        json.field "query_labels", fixture.query_labels
        json.field "key_labels", fixture.key_labels
        json.field "named_metal_bytes_after_setup", setup_named_bytes
        json.field "named_buffer_allocations_during_encode", encode_allocations
        json.field "encode_readbacks_before_terminal_wait", preterminal_readbacks
        json.field "waits_before_terminal_wait", preterminal_wait_count
        json.field "terminal_wait_count", terminal_wait_count
        json.field "diagnostic_readbacks_after_wait", diagnostic_readbacks
        json.field "early_masked_plan_release_rejected", early_masked_plan_release_rejected
        json.field "early_legacy_plan_release_rejected", early_legacy_plan_release_rejected
        json.field "early_workspace_release_rejected", early_workspace_release_rejected
        json.field "max_masked_vs_legacy_active_output_delta", active_delta
        json.field "mask_changes_output", mask_changes_output
        json.field "encode_seconds", encode_seconds
        json.field "terminal_wait_seconds", wait_seconds
        json.field "gpu_elapsed_seconds", chain.not_nil!.gpu_elapsed_seconds
        json.field "tolerance_atol", HyNomasActionAttentionProbe::ATOL
        json.field "comparisons" do
          HyNomasActionAttentionProbe.compare_json(json, comparisons)
        end
        json.field "passed", !failed && constraints_passed
      end
    end
  end
  puts report_path
  comparisons.each do |row|
    row_json = JSON.build do |json|
      json.object do
        json.field "name", row.name
        json.field "max_abs", row.max_abs
        json.field "failures", row.failures
        json.field "first_bad_index", row.first_bad_index
        json.field "finite", row.finite
      end
    end
    puts row_json
  end
  raise "action-attention native probe failed numerical or execution constraints" if failed || !constraints_passed
  puts "PASS: source-range action mask and unlabelled legacy control; one terminal wait"
ensure
  if active_chain = chain
    active_chain.commit_and_wait unless active_chain.completed?
  end
  masked_plan.try(&.release)
  legacy_plan.try(&.release)
  masked_workspace.try(&.release)
  legacy_workspace.try(&.release)
  chain.try(&.release)
  {
    query_buffer, masked_current_key_buffer, masked_current_value_buffer,
    legacy_current_key_buffer, legacy_current_value_buffer, history_key_buffer,
    history_value_buffer, masked_output, legacy_output,
  }.each { |buffer| buffer.try(&.release) }
end
