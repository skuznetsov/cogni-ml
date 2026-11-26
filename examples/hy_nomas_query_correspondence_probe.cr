require "digest/sha256"
require "json"
require "option_parser"
require "../src/ml/video/metal_ops"
require "../src/ml/video/metal_mps_command_chain"
require "./support/hy_nomas_probe_instrumentation"
require "./support/hy_nomas_query_correspondence_probe"

manifest_path = ""
manifest_sha256 = ""
report_path = ""
execute_gpu = false
admit_only = false
OptionParser.parse do |parser|
  parser.banner = "Screen F32 Q/K self-correspondence from authenticated BF16-expanded block-0 tensors."
  parser.on("--manifest PATH", "Authenticated query/key input manifest") { |value| manifest_path = value }
  parser.on("--manifest-sha256 SHA", "Trusted input manifest SHA-256") { |value| manifest_sha256 = value }
  parser.on("--report PATH", "Fresh JSON report path; a sibling .tensors.f32le payload is created") { |value| report_path = value }
  parser.on("--execute", "Explicitly allow the two native Metal diagnostic kernels") { execute_gpu = true }
  parser.on("--admit-only", "Authenticate and validate inputs without initializing Metal") { admit_only = true }
end

unless ARGV.empty? && !manifest_path.empty? && !manifest_sha256.empty? && !report_path.empty? &&
       execute_gpu != admit_only
  raise ArgumentError.new("manifest, trusted SHA-256, report, and exactly one of --execute or --admit-only are required")
end

report_path = File.expand_path(report_path)
unless report_path.ends_with?(".json") && Dir.exists?(File.dirname(report_path)) &&
       File.realpath(File.dirname(report_path)) == File.dirname(report_path)
  raise ArgumentError.new("report must be a .json file in a canonical existing directory")
end
payload_path = report_path.sub(/\.json\z/, ".tensors.f32le")
{report_path, payload_path}.each do |path|
  if File.exists?(path) || File.symlink?(path)
    raise ArgumentError.new("refusing to replace output: #{path}")
  end
end

# Complete all manifest, file, range, digest, dtype, and finite-value checks
# before Arena initialization, pipeline compilation, or command-chain creation.
fixture = HyNomasQueryCorrespondenceProbe::Fixture.new(manifest_path, manifest_sha256)
if admit_only
  admission_json = JSON.build do |json|
    json.object do
      json.field "schema", HyNomasQueryCorrespondenceProbe::INPUT_SCHEMA
      json.field "manifest_path", fixture.manifest_path
      json.field "manifest_sha256", fixture.manifest_sha256
      json.field "payload_path", fixture.payload_path
      json.field "payload_sha256", fixture.payload_sha256
      json.field "geometry", {heads:     HyNomasQueryCorrespondenceProbe::HEADS,
                              positions: HyNomasQueryCorrespondenceProbe::POSITIONS,
                              head_dim:  HyNomasQueryCorrespondenceProbe::HEAD_DIM}
      json.field "metal_initialized", false
      json.field "admitted", true
    end
  end
  puts admission_json
  puts "PASS: authenticated query/key manifest admitted without Metal initialization"
  exit(0)
end

include HyNomasQueryCorrespondenceProbe
arena : ML::Video::Arena? = nil
chain : ML::Metal::MPSCommandChain? = nil
begin
  arena = ML::Video::Arena.new
  query = arena.not_nil!.alloc("query_correspondence.input_query", TENSOR_VALUES)
  key = arena.not_nil!.alloc("query_correspondence.input_key", TENSOR_VALUES)
  scores = arena.not_nil!.alloc("query_correspondence.scores_f32", SCORE_VALUES)
  metrics = arena.not_nil!.alloc("query_correspondence.metrics_f32", METRIC_COUNT * METRIC_WIDTH)
  query.write_bytes(fixture.query_bytes.to_unsafe, fixture.query_bytes.bytesize)
  key.write_bytes(fixture.key_bytes.to_unsafe, fixture.key_bytes.bytesize)
  named_bytes = arena.not_nil!.resident_bytes
  unless named_bytes == NAMED_BYTES
    raise ArgumentError.new("query-correspondence named buffer inventory or byte count mismatch")
  end

  score_pipeline = ML::Metal::PipelineCache.get_or_compile(
    "hy_nomas_query_correspondence_scores_f32", METAL_SOURCE)
  metric_pipeline = ML::Metal::PipelineCache.get_or_compile(
    "hy_nomas_query_correspondence_metrics_f32", METAL_SOURCE)
  chain = ML::Metal::MPSCommandChain.new
  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
  waits_before = chain.not_nil!.wait_count

  encode_started = Time.instant
  encoder = chain.not_nil!.encoder
  encoder.set_pipeline(score_pipeline).set_buffer(query, 0).set_buffer(key, 1)
    .set_buffer(scores, 2, ML::Metal::BufferAccess::Write)
    .dispatch_1d(SCORE_VALUES)
  encoder.set_pipeline(metric_pipeline).set_buffer(scores, 0)
    .set_buffer(metrics, 1, ML::Metal::BufferAccess::Write)
    .dispatch_1d(METRIC_COUNT)
  encode_seconds = (Time.instant - encode_started).total_seconds
  named_encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  encode_waits = chain.not_nil!.wait_count - waits_before
  unless named_encode_allocations == 0 && encode_readbacks == 0 && encode_waits == 0
    raise ArgumentError.new("query-correspondence encode allocated, read back, or waited")
  end

  wait_started = Time.instant
  chain.not_nil!.commit_and_wait
  wait_seconds = (Time.instant - wait_started).total_seconds
  terminal_waits = chain.not_nil!.wait_count
  raise ArgumentError.new("query-correspondence requires exactly one terminal wait") unless terminal_waits == 1

  score_values = scores.read(SCORE_VALUES)
  metric_values = metrics.read(METRIC_COUNT * METRIC_WIDTH)
  terminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  raise ArgumentError.new("query-correspondence requires two post-wait diagnostic readbacks") unless terminal_readbacks == 2
  score_bytes = Bytes.new(score_values.to_unsafe.as(UInt8*), score_values.size * sizeof(Float32))
  metric_bytes = Bytes.new(metric_values.to_unsafe.as(UInt8*), metric_values.size * sizeof(Float32))
  score_sha256 = Digest::SHA256.hexdigest(score_bytes)
  metric_sha256 = Digest::SHA256.hexdigest(metric_bytes)
  payload_digest = Digest::SHA256.new
  payload_digest.update(score_bytes)
  payload_digest.update(metric_bytes)
  output_payload_sha256 = payload_digest.hexfinal
  output_payload_bytes = score_bytes.size.to_i64 + metric_bytes.size.to_i64
  unless output_payload_bytes == SCORE_BYTES + METRIC_BYTES
    raise ArgumentError.new("query-correspondence output payload byte count mismatch")
  end

  gpu_elapsed_seconds = chain.not_nil!.gpu_elapsed_seconds
  chain.not_nil!.release
  chain = nil
  arena.not_nil!.release
  arena = nil

  HyNomasQueryCorrespondenceProbe.write_fresh_parts(payload_path, [score_bytes, metric_bytes])
  report_json = JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", OUTPUT_SCHEMA
      json.field "schema_version", 1
      json.field "scope", "F32 QK score and row-statistic diagnostic over authenticated F32 values exactly expanded from BF16 query/key tensors; it screens self-correspondence only and is not BF16 attention parity or evidence of acting"
      json.field "input_manifest_path", fixture.manifest_path
      json.field "input_manifest_sha256", fixture.manifest_sha256
      json.field "input_payload_sha256", fixture.payload_sha256
      if provenance = fixture.provenance
        json.field "provenance", provenance
      end
      json.field "geometry", {heads: HEADS, positions: POSITIONS, head_dim: HEAD_DIM,
                              layout: "[positions,heads,head_dim]", score_scale: "1/sqrt(128)"}
      json.field "input_tensors" do
        json.object do
          {"query" => fixture.query_sha256, "key" => fixture.key_sha256}.each do |name, digest|
            json.field name do
              json.object do
                json.field "dtype", "F32"
                json.field "source_dtype", "BF16"
                json.field "shape", [POSITIONS, HEADS, HEAD_DIM]
                json.field "sha256", digest
              end
            end
          end
        end
      end
      json.field "payload" do
        json.object do
          json.field "path", File.basename(payload_path)
          json.field "bytes", output_payload_bytes
          json.field "sha256", output_payload_sha256
          json.field "byte_order", "little"
        end
      end
      json.field "tensors" do
        json.object do
          json.field "scores" do
            json.object do
              json.field "dtype", "F32"
              json.field "shape", [HEADS, POSITIONS, POSITIONS]
              json.field "offset_bytes", 0
              json.field "nbytes", SCORE_BYTES
              json.field "sha256", score_sha256
            end
          end
          json.field "metrics" do
            json.object do
              json.field "dtype", "F32"
              json.field "shape", [HEADS, POSITIONS, METRIC_WIDTH]
              json.field "offset_bytes", SCORE_BYTES
              json.field "nbytes", METRIC_BYTES
              json.field "sha256", metric_sha256
            end
          end
        end
      end
      json.field "metrics_order", METRICS_ORDER
      json.field "execution" do
        json.object do
          json.field "device", "metal"
          json.field "named_bytes_after_setup", named_bytes
          json.field "named_encode_allocations", named_encode_allocations
          json.field "encode_readbacks", encode_readbacks
          json.field "encode_waits", encode_waits
          json.field "terminal_waits", terminal_waits
          json.field "terminal_readbacks", terminal_readbacks
          json.field "encode_seconds", encode_seconds
          json.field "terminal_wait_seconds", wait_seconds
          json.field "gpu_elapsed_seconds", gpu_elapsed_seconds
        end
      end
    end
  end
  report_json += "\n"
  HyNomasQueryCorrespondenceProbe.write_fresh_bytes(report_path, report_json.to_slice)
  puts report_path
  puts "PASS: F32 QK scores and correspondence metrics captured after one terminal wait"
ensure
  if active_chain = chain
    begin
      active_chain.commit_and_wait unless active_chain.completed?
    ensure
      active_chain.release if active_chain.completed?
    end
  end
  if active_arena = arena
    active_arena.release if chain.nil? || chain.not_nil!.completed?
  end
end
