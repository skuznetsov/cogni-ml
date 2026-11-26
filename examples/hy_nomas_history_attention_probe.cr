require "digest"
require "json"
require "option_parser"
require "../src/ml/video/hy_nomas_attention_compact"
require "../src/ml/video/hy_nomas_history_attention"
require "./support/hy_nomas_history_attention_probe_guard"
require "./support/hy_nomas_probe_instrumentation"

module HyNomasHistoryAttentionProbe
  UPSTREAM_REVISION      = "911cea6458a89c2663513ff7099561ec190dd491"
  UPSTREAM_SOURCE_SHA256 = "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee"
  QUERY_ROWS             = 257_i32
  HISTORY_ROWS           =  23_i32
  HEADS                  =  16_i32
  HEAD_DIM               = 128_i32
  ELEMENTS               = QUERY_ROWS.to_i64 * HEADS * HEAD_DIM
  OUTPUT_BYTES           = ELEMENTS * sizeof(Float32)
  ATOL                   = 2.0e-5

  record Comparison, name : String, max_abs : Float64?, failures : Int32,
    first_bad_index : Int32?, finite : Bool

  class Fixture
    getter manifest_sha256 : String
    getter query : String
    getter current_key : String
    getter current_value : String
    getter history_key : String
    getter history_value : String
    getter expected_p0 : Array(Float32)
    getter expected_p23 : Array(Float32)

    def initialize(manifest_path : String, expected_manifest_sha256 : String,
                   oracle_source_path : String, upstream_source_path : String)
      unless expected_manifest_sha256.matches?(/\A[0-9a-fA-F]{64}\z/)
        raise ArgumentError.new("fixture manifest SHA-256 must be 64 hex characters")
      end
      raise ArgumentError.new("fixture manifest cannot be a symlink") if File.symlink?(manifest_path)
      raw_manifest = File.read(manifest_path)
      @manifest_sha256 = Digest::SHA256.hexdigest(raw_manifest)
      unless @manifest_sha256 == expected_manifest_sha256.downcase
        raise ArgumentError.new("fixture manifest digest mismatch")
      end

      document = JSON.parse(raw_manifest)
      raise ArgumentError.new("unexpected history-attention fixture schema") unless document["schema"].as_s == "cogni_ml.hy_nomas_history_attention_oracle"
      raise ArgumentError.new("unexpected history-attention fixture version") unless document["schema_version"].as_i == 1

      upstream = document["upstream"]
      raise ArgumentError.new("upstream revision pin mismatch") unless upstream["revision"].as_s == UPSTREAM_REVISION
      raise ArgumentError.new("upstream source path pin mismatch") unless upstream["source_file"].as_s == "hymm/models/dit/nomas_cpt.py"
      raise ArgumentError.new("upstream equation pin mismatch") unless upstream["route"].as_s.includes?("attn_mode == 'torch'")
      unless upstream["source_file_sha256"].as_s == UPSTREAM_SOURCE_SHA256
        raise ArgumentError.new("upstream source digest pin mismatch")
      end
      raise ArgumentError.new("upstream source cannot be a symlink") if File.symlink?(upstream_source_path)
      actual_upstream_sha = Digest::SHA256.hexdigest(File.read(upstream_source_path))
      unless actual_upstream_sha == UPSTREAM_SOURCE_SHA256
        raise ArgumentError.new("local upstream source does not match pinned nomas_cpt.py")
      end

      oracle = document["oracle"]
      raise ArgumentError.new("oracle must be CPU-only") unless oracle["device"].as_s == "cpu"
      raise ArgumentError.new("oracle must use F32") unless oracle["dtype"].as_s == "torch.float32"
      raise ArgumentError.new("oracle source cannot be a symlink") if File.symlink?(oracle_source_path)
      actual_oracle_sha = Digest::SHA256.hexdigest(File.read(oracle_source_path))
      unless actual_oracle_sha == oracle["script_sha256"].as_s
        raise ArgumentError.new("oracle source does not match authenticated fixture metadata")
      end

      geometry = document["geometry"]
      expected_geometry = {
        "batch"        => 1_i64,
        "query_rows"   => QUERY_ROWS.to_i64,
        "history_rows" => HISTORY_ROWS.to_i64,
        "heads"        => HEADS.to_i64,
        "head_dim"     => HEAD_DIM.to_i64,
      }
      expected_geometry.each do |key, expected|
        raise ArgumentError.new("fixture geometry mismatch for #{key}") unless geometry[key].as_i == expected
      end
      raise ArgumentError.new("fixture tensors must be little-endian") unless document["tensors"]["query"]["byte_order"].as_s == "little"

      directory = File.dirname(File.expand_path(manifest_path))
      tensors = document["tensors"]
      @query = read_tensor(directory, tensors["query"], ELEMENTS)
      @current_key = read_tensor(directory, tensors["current_key"], ELEMENTS)
      @current_value = read_tensor(directory, tensors["current_value"], ELEMENTS)
      history_elements = HISTORY_ROWS.to_i64 * HEADS * HEAD_DIM
      @history_key = read_tensor(directory, tensors["history_key"], history_elements)
      @history_value = read_tensor(directory, tensors["history_value"], history_elements)

      cases = document["cases"].as_a
      p0 = cases.find { |entry| entry["history_rows"].as_i == 0_i64 }
      p23 = cases.find { |entry| entry["history_rows"].as_i == HISTORY_ROWS.to_i64 }
      raise ArgumentError.new("fixture must include exactly P=0 and P=23 cases") unless cases.size == 2 && p0 && p23
      @expected_p0 = decode_f32(read_tensor(directory, p0.not_nil!["expected"], ELEMENTS))
      @expected_p23 = decode_f32(read_tensor(directory, p23.not_nil!["expected"], ELEMENTS))
    end

    private def read_tensor(directory : String, descriptor : JSON::Any,
                            expected_elements : Int64) : String
      raise ArgumentError.new("fixture tensor dtype must be F32") unless descriptor["dtype"].as_s == "F32"
      raise ArgumentError.new("fixture tensor byte order must be little-endian") unless descriptor["byte_order"].as_s == "little"
      filename = descriptor["file"].as_s
      unless File.basename(filename) == filename && filename != "." && filename != ".."
        raise ArgumentError.new("fixture tensor path must be a basename")
      end
      path = File.join(directory, filename)
      raise ArgumentError.new("fixture tensor cannot be a symlink") if File.symlink?(path)
      raw = File.read(path)
      expected_bytes = expected_elements * sizeof(Float32)
      raise ArgumentError.new("fixture tensor byte count mismatch for #{filename}") unless raw.bytesize.to_i64 == expected_bytes && descriptor["nbytes"].as_i == expected_bytes
      unless Digest::SHA256.hexdigest(raw) == descriptor["sha256"].as_s
        raise ArgumentError.new("fixture tensor digest mismatch for #{filename}")
      end
      shape = descriptor["shape"].as_a.map(&.as_i)
      expected_shapes = [
        {1_i64, QUERY_ROWS.to_i64, HEADS.to_i64, HEAD_DIM.to_i64},
        {1_i64, HISTORY_ROWS.to_i64, HEADS.to_i64, HEAD_DIM.to_i64},
      ]
      raise ArgumentError.new("fixture tensor shape has wrong element count") unless shape.reduce(1_i64) { |product, dimension| product * dimension } == expected_elements
      raise ArgumentError.new("fixture tensor has an unsupported rank") unless shape.size == 4 && expected_shapes.includes?({shape[0], shape[1], shape[2], shape[3]})
      raw
    end

    private def decode_f32(raw : String) : Array(Float32)
      unless raw.bytesize % sizeof(Float32) == 0
        raise ArgumentError.new("fixture F32 sidecar length is not aligned")
      end
      bytes = raw.to_slice
      values = bytes.to_unsafe.as(Pointer(Float32))
      Array(Float32).new(raw.bytesize // sizeof(Float32)) { |index| values[index] }
    end
  end

  def self.upload_f32(raw : String) : ML::MetalBuffer
    buffer = ML::MetalBuffer.new(raw.bytesize.to_i64)
    buffer.write_bytes(raw.to_unsafe.as(Pointer(UInt8)), raw.bytesize)
    buffer
  end

  def self.output_buffer : ML::MetalBuffer
    ML::MetalBuffer.new(OUTPUT_BYTES)
  end

  def self.compare(name : String, actual : Array(Float32),
                   expected : Array(Float32)) : Comparison
    raise ArgumentError.new("comparison vector length mismatch: #{name}") unless actual.size == expected.size
    max_abs = 0.0_f64
    finite = true
    failures = 0_i32
    first_bad = nil.as(Int32?)
    actual.each_with_index do |value, index|
      reference = expected[index]
      unless value.finite? && reference.finite?
        finite = false
        failures += 1
        first_bad ||= index.to_i32
        next
      end
      error = (value.to_f64 - reference.to_f64).abs
      max_abs = Math.max(max_abs, error)
      if error > ATOL
        failures += 1
        first_bad ||= index.to_i32
      end
    end
    Comparison.new(name, finite ? max_abs : nil, failures, first_bad, finite)
  end

  def self.compare_json(json : JSON::Builder, rows : Array(Comparison)) : Nil
    json.array do
      rows.each do |row|
        json.object do
          json.field "name", row.name
          json.field "max_abs", row.max_abs
          json.field "failures", row.failures
          json.field "first_bad_index", row.first_bad_index
          json.field "finite", row.finite
        end
      end
    end
  end
end

fixture_path = ""
fixture_sha256 = ""
oracle_source = ""
upstream_source = ""
report_path = ""
run_native = false
validate_fixture_only = false
OptionParser.parse do |parser|
  parser.banner = "Opt-in synthetic F32 history-attention equivalence probe; no model weights or generation."
  parser.on("--fixture-manifest PATH", "CPU oracle manifest") { |value| fixture_path = value }
  parser.on("--fixture-sha256 SHA256", "Trusted CPU oracle manifest digest") { |value| fixture_sha256 = value }
  parser.on("--oracle-source PATH", "CPU oracle Python source whose hash is in the manifest") { |value| oracle_source = value }
  parser.on("--upstream-source PATH", "Pinned upstream nomas_cpt.py source") { |value| upstream_source = value }
  parser.on("--report PATH", "New post-terminal JSON report") { |value| report_path = value }
  parser.on("--run-native", "Explicitly allow the scheduled native Metal probe") { run_native = true }
  parser.on("--validate-fixture-only", "Validate authenticated CPU fixture inputs without creating a Metal device") { validate_fixture_only = true }
end
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
raise ArgumentError.new("choose either --validate-fixture-only or --run-native") if validate_fixture_only && run_native
if validate_fixture_only
  raise ArgumentError.new("fixture, trusted digest, oracle source and upstream source are required") if fixture_path.empty? || fixture_sha256.empty? || oracle_source.empty? || upstream_source.empty?
  fixture = HyNomasHistoryAttentionProbe::Fixture.new(fixture_path, fixture_sha256, oracle_source, upstream_source)
  puts "PASS: authenticated P=0/P=23 CPU fixture validated; no Metal device created"
  exit(0)
end
raise ArgumentError.new("--run-native is required; the probe is intentionally opt-in") unless run_native
raise ArgumentError.new("fixture, trusted digest, oracle source and upstream source are required") if fixture_path.empty? || fixture_sha256.empty? || oracle_source.empty? || upstream_source.empty?
raise ArgumentError.new("native report path is required") if report_path.empty?
raise ArgumentError.new("refusing to replace native report") if File.exists?(report_path) || File.symlink?(report_path)

fixture = HyNomasHistoryAttentionProbe::Fixture.new(fixture_path, fixture_sha256, oracle_source, upstream_source)
query_buffer : ML::MetalBuffer? = nil
current_key_buffer : ML::MetalBuffer? = nil
current_value_buffer : ML::MetalBuffer? = nil
history_key_buffer : ML::MetalBuffer? = nil
history_value_buffer : ML::MetalBuffer? = nil
p0_output : ML::MetalBuffer? = nil
square_output : ML::MetalBuffer? = nil
p23_output : ML::MetalBuffer? = nil
p0_workspace : ML::Video::HyNomasHistoryAttentionWorkspace? = nil
p23_workspace : ML::Video::HyNomasHistoryAttentionWorkspace? = nil
p0_plan : ML::Video::HyNomasHistoryAttentionMPSPlan? = nil
p23_plan : ML::Video::HyNomasHistoryAttentionMPSPlan? = nil
square_plan : ML::Video::HyNomasCompactAttentionMPSPlan? = nil
chain : ML::Metal::MPSCommandChain? = nil
initial_named_bytes = ML::MetalBuffer.stats[:live_bytes]
comparisons = [] of HyNomasHistoryAttentionProbe::Comparison
encode_allocations = 0_i64
preterminal_readbacks = 0_i64
terminal_wait_count = 0_i32
diagnostic_readbacks = 0_i64
setup_named_bytes = 0_i64
early_plan_release_rejected = false
early_square_release_rejected = false
early_p23_plan_release_rejected = false
early_workspace_release_rejected = false
encode_seconds = 0.0
wait_seconds = 0.0
begin
  query_buffer = HyNomasHistoryAttentionProbe.upload_f32(fixture.query)
  current_key_buffer = HyNomasHistoryAttentionProbe.upload_f32(fixture.current_key)
  current_value_buffer = HyNomasHistoryAttentionProbe.upload_f32(fixture.current_value)
  history_key_buffer = HyNomasHistoryAttentionProbe.upload_f32(fixture.history_key)
  history_value_buffer = HyNomasHistoryAttentionProbe.upload_f32(fixture.history_value)
  p0_output = HyNomasHistoryAttentionProbe.output_buffer
  square_output = HyNomasHistoryAttentionProbe.output_buffer
  p23_output = HyNomasHistoryAttentionProbe.output_buffer

  p0_workspace = ML::Video::HyNomasHistoryAttentionWorkspace.new(0, HyNomasHistoryAttentionProbe::QUERY_ROWS,
    HyNomasHistoryAttentionProbe::HEADS, HyNomasHistoryAttentionProbe::HEAD_DIM)
  p0_workspace.not_nil!.freeze_history_inventory!
  p23_workspace = ML::Video::HyNomasHistoryAttentionWorkspace.new(HyNomasHistoryAttentionProbe::HISTORY_ROWS,
    HyNomasHistoryAttentionProbe::QUERY_ROWS, HyNomasHistoryAttentionProbe::HEADS,
    HyNomasHistoryAttentionProbe::HEAD_DIM)
  history_slot = p23_workspace.not_nil!.register_history_pair!(history_key_buffer.not_nil!, history_value_buffer.not_nil!)
  p23_workspace.not_nil!.freeze_history_inventory!

  p0_plan = ML::Video::HyNomasHistoryAttentionMPSPlan.new(p0_workspace.not_nil!, query_buffer.not_nil!,
    current_key_buffer.not_nil!, current_value_buffer.not_nil!, p0_output.not_nil!)
  p23_plan = ML::Video::HyNomasHistoryAttentionMPSPlan.new(p23_workspace.not_nil!, query_buffer.not_nil!,
    current_key_buffer.not_nil!, current_value_buffer.not_nil!, p23_output.not_nil!)
  square_plan = ML::Video::HyNomasCompactAttentionMPSPlan.new(1,
    HyNomasHistoryAttentionProbe::QUERY_ROWS, HyNomasHistoryAttentionProbe::HEADS,
    HyNomasHistoryAttentionProbe::HEAD_DIM, query_buffer.not_nil!, current_key_buffer.not_nil!,
    current_value_buffer.not_nil!, square_output.not_nil!, 256, HyNomasHistoryAttentionProbe::HEADS)

  p0_geometry = p0_workspace.not_nil!.shape
  p23_geometry = p23_workspace.not_nil!.shape
  setup_named_bytes = ML::MetalBuffer.stats[:live_bytes] - initial_named_bytes
  chain = ML::Metal::MPSCommandChain.new

  before_allocations = ML::MetalBuffer.hy_text_probe_allocations
  before_readbacks = ML::MetalBuffer.hy_text_probe_readbacks
  started = Time.instant
  p0_plan.not_nil!.encode(chain.not_nil!)
  square_plan.not_nil!.encode(chain.not_nil!)
  p23_plan.not_nil!.encode(chain.not_nil!, history_slot)
  encode_seconds = (Time.instant - started).total_seconds
  encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - before_allocations
  preterminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - before_readbacks
  preterminal_wait_count = chain.not_nil!.wait_count

  begin
    p0_plan.not_nil!.release
  rescue ex : ArgumentError
    early_plan_release_rejected = HyNomasHistoryAttentionProbe.inflight_chain_release_rejected?(
      chain.not_nil!.completed?, ex.message)
  end
  begin
    square_plan.not_nil!.release
  rescue ex : ArgumentError
    early_square_release_rejected = HyNomasHistoryAttentionProbe.inflight_chain_release_rejected?(
      chain.not_nil!.completed?, ex.message)
  end
  begin
    p23_plan.not_nil!.release
  rescue ex : ArgumentError
    early_p23_plan_release_rejected = HyNomasHistoryAttentionProbe.inflight_chain_release_rejected?(
      chain.not_nil!.completed?, ex.message)
  end
  begin
    p23_workspace.not_nil!.release
  rescue ex : ArgumentError
    early_workspace_release_rejected = ex.message.try(&.includes?("Release all")) || false
  end
  raise "in-flight attention plan release was not rejected" unless early_plan_release_rejected && early_square_release_rejected && early_p23_plan_release_rejected
  raise "in-flight workspace release was not rejected" unless early_workspace_release_rejected

  started = Time.instant
  chain.not_nil!.commit_and_wait
  wait_seconds = (Time.instant - started).total_seconds
  terminal_wait_count = chain.not_nil!.wait_count
  raise "expected one terminal wait" unless terminal_wait_count == 1

  actual_p0 = p0_output.not_nil!.read(HyNomasHistoryAttentionProbe::ELEMENTS.to_i32)
  actual_square = square_output.not_nil!.read(HyNomasHistoryAttentionProbe::ELEMENTS.to_i32)
  actual_p23 = p23_output.not_nil!.read(HyNomasHistoryAttentionProbe::ELEMENTS.to_i32)
  diagnostic_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - before_readbacks
  comparisons = [
    HyNomasHistoryAttentionProbe.compare("P0 native history path vs pinned upstream CPU F32 oracle", actual_p0, fixture.expected_p0),
    HyNomasHistoryAttentionProbe.compare("P0 compact square path vs pinned upstream CPU F32 oracle", actual_square, fixture.expected_p0),
    HyNomasHistoryAttentionProbe.compare("P0 history path vs compact square path", actual_p0, actual_square),
    HyNomasHistoryAttentionProbe.compare("P23 rectangular history path vs pinned upstream CPU F32 oracle", actual_p23, fixture.expected_p23),
  ]

  failed = comparisons.any? { |row| row.failures > 0 || !row.finite }
  constraints_passed = encode_allocations == 0 && preterminal_readbacks == 0 &&
                       preterminal_wait_count == 0 && terminal_wait_count == 1 && diagnostic_readbacks == 3
  File.open(report_path, "w") do |file|
    JSON.build(file, indent: "  ") do |json|
      json.object do
        json.field "schema", "cogni_ml.hy_nomas_history_attention_native_probe"
        json.field "schema_version", 1
        json.field "fixture_manifest", File.expand_path(fixture_path)
        json.field "fixture_manifest_sha256", fixture.manifest_sha256
        json.field "upstream_revision", HyNomasHistoryAttentionProbe::UPSTREAM_REVISION
        json.field "upstream_source_sha256", HyNomasHistoryAttentionProbe::UPSTREAM_SOURCE_SHA256
        json.field "oracle_source_sha256", Digest::SHA256.hexdigest(File.read(oracle_source))
        json.field "input_scope", "deterministic synthetic F32 Q/K/V; same Q/K/V files used for P=0 and P=23; no model weights or generation"
        json.field "precision_scope", "all 16 heads; F32 QK, softmax, PV; no padding keys; history prefix is P=23"
        json.field "counter_scope", "ML::MetalBuffer named allocations and reads; opaque Objective-C/MPSGraph internal resources are not counted"
        json.field "geometry", {batch: 1, query_rows: HyNomasHistoryAttentionProbe::QUERY_ROWS,
                                history_rows: HyNomasHistoryAttentionProbe::HISTORY_ROWS, heads: HyNomasHistoryAttentionProbe::HEADS,
                                head_dim: HyNomasHistoryAttentionProbe::HEAD_DIM, p0_graph_calls: p0_geometry.graph_calls,
                                p23_graph_calls: p23_geometry.graph_calls, query_chunk: p23_geometry.query_chunk,
                                p0_workspace_bytes: p0_geometry.workspace_bytes, p23_workspace_bytes: p23_geometry.workspace_bytes,
                                p23_max_score_tile_bytes: p23_geometry.max_score_bytes}
        json.field "named_metal_bytes_after_setup", setup_named_bytes
        json.field "named_buffer_allocations_during_encode", encode_allocations
        json.field "encode_readbacks_before_terminal_wait", preterminal_readbacks
        json.field "waits_before_terminal_wait", preterminal_wait_count
        json.field "terminal_wait_count", terminal_wait_count
        json.field "diagnostic_readbacks_after_wait", diagnostic_readbacks
        json.field "early_plan_release_rejected", early_plan_release_rejected
        json.field "early_square_release_rejected", early_square_release_rejected
        json.field "early_p23_plan_release_rejected", early_p23_plan_release_rejected
        json.field "early_workspace_release_rejected", early_workspace_release_rejected
        json.field "encode_seconds", encode_seconds
        json.field "terminal_wait_seconds", wait_seconds
        json.field "gpu_elapsed_seconds", chain.not_nil!.gpu_elapsed_seconds
        json.field "tolerance_atol", HyNomasHistoryAttentionProbe::ATOL
        json.field "comparisons" do
          HyNomasHistoryAttentionProbe.compare_json(json, comparisons)
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
  raise "history-attention native probe failed numerical or execution constraints" if failed || !constraints_passed
  puts "PASS: P=0 square parity and P=23 rectangular F32 history against pinned CPU oracle; one terminal wait"
ensure
  if active_chain = chain
    active_chain.commit_and_wait unless active_chain.completed?
  end
  p0_plan.try(&.release)
  square_plan.try(&.release)
  p23_plan.try(&.release)
  p0_workspace.try(&.release)
  p23_workspace.try(&.release)
  chain.try(&.release)
  {query_buffer, current_key_buffer, current_value_buffer, history_key_buffer,
   history_value_buffer, p0_output, square_output, p23_output}.each do |buffer|
    buffer.try(&.release)
  end
end
