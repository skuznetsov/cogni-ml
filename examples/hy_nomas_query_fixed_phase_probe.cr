require "digest/sha256"
require "json"
require "option_parser"
require "../src/ml/video/metal_ops"
require "../src/ml/video/metal_mps_command_chain"
require "../src/ml/video/hy_nomas_block"
require "../src/ml/video/hy_nomas_mps_pointwise"
require "./support/hy_nomas_probe_instrumentation"
require "./support/hy_nomas_query_correspondence_probe"
require "./support/hy_nomas_query_fixed_phase_probe"

manifest_path = ""
manifest_sha256 = ""
report_path = ""
execute_gpu = false
admit_only = false
OptionParser.parse do |parser|
  parser.banner = "Run a fixed-phase key-content null on authenticated block-0 Q/K."
  parser.on("--manifest PATH", "Authenticated fixed-phase input manifest") { |value| manifest_path = value }
  parser.on("--manifest-sha256 SHA", "Trusted input manifest SHA-256") { |value| manifest_sha256 = value }
  parser.on("--report PATH", "Fresh JSON report path; a sibling .tensors.f32le payload is created") { |value| report_path = value }
  parser.on("--execute", "Explicitly allow the native Metal diagnostic kernels") { execute_gpu = true }
  parser.on("--admit-only", "Authenticate inputs without initializing Metal") { admit_only = true }
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

# Authenticate Q/K and selected finite F32 phase rows before Metal allocation,
# pipeline compilation, or command-chain creation.
fixture = HyNomasQueryFixedPhaseProbe::Fixture.new(manifest_path, manifest_sha256)
if admit_only
  admission_json = JSON.build do |json|
    json.object do
      json.field "schema", HyNomasQueryFixedPhaseProbe::INPUT_SCHEMA
      json.field "manifest_path", fixture.manifest_path
      json.field "manifest_sha256", fixture.manifest_sha256
      json.field "qk_manifest_sha256", fixture.qk_manifest_sha256
      json.field "phase_payload_sha256", fixture.phase_payload_sha256
      json.field "case" do
        json.object do
          json.field "frame", fixture.frame
          json.field "arm", fixture.arm
          json.field "key_roll", fixture.key_roll
        end
      end
      json.field "geometry", {heads:     HyNomasQueryFixedPhaseProbe::HEADS,
                              positions: HyNomasQueryFixedPhaseProbe::POSITIONS,
                              head_dim:  HyNomasQueryFixedPhaseProbe::HEAD_DIM}
      json.field "metal_initialized", false
      json.field "admitted", true
    end
  end
  puts admission_json
  puts "PASS: fixed-phase input admitted without Metal initialization"
  exit(0)
end

include HyNomasQueryFixedPhaseProbe

arena : ML::Video::Arena? = nil
chain : ML::Metal::MPSCommandChain? = nil
begin
  arena = ML::Video::Arena.new
  query = arena.not_nil!.alloc("fixed_phase.input_query", TENSOR_VALUES)
  key = arena.not_nil!.alloc("fixed_phase.input_key", TENSOR_VALUES)
  cosine = arena.not_nil!.alloc("fixed_phase.cosine", PHASE_VALUES)
  sine = arena.not_nil!.alloc("fixed_phase.sine", PHASE_VALUES)
  work_key = arena.not_nil!.alloc("fixed_phase.work_key", TENSOR_VALUES)
  work_rope_key = arena.not_nil!.alloc("fixed_phase.work_rope_key", TENSOR_VALUES)
  output_query = arena.not_nil!.alloc("fixed_phase.output_query", TENSOR_VALUES)
  output_key = arena.not_nil!.alloc("fixed_phase.output_key", TENSOR_VALUES)
  scores = arena.not_nil!.alloc("fixed_phase.scores_f32", SCORE_VALUES)
  metrics = arena.not_nil!.alloc("fixed_phase.metrics_f32", METRIC_COUNT * METRIC_WIDTH)
  query.write_bytes(fixture.query_bytes.to_unsafe, fixture.query_bytes.bytesize)
  key.write_bytes(fixture.key_bytes.to_unsafe, fixture.key_bytes.bytesize)
  cosine.write_bytes(fixture.phase_cosine_bytes.to_unsafe, fixture.phase_cosine_bytes.bytesize)
  sine.write_bytes(fixture.phase_sine_bytes.to_unsafe, fixture.phase_sine_bytes.bytesize)

  named_bytes = arena.not_nil!.resident_bytes
  unless named_bytes == NAMED_BYTES && arena.not_nil!.buffers.size == BUFFER_COUNT
    raise ArgumentError.new("fixed-phase named buffer inventory or byte count mismatch")
  end
  resident_buffers = arena.not_nil!.buffers.values
  resident_buffers.each_with_index do |buffer, index|
    resident_buffers[(index + 1)..].each do |other|
      if ML::Video::MetalOps.overlap?(buffer, other)
        raise ArgumentError.new("fixed-phase named buffers alias or overlap")
      end
    end
  end

  rope_pipeline = ML::Metal::PipelineCache.get_or_compile(
    ML::Video::HyNomasMPSPointwise::ROPE_KERNEL, METAL_SOURCE)
  gather_pipeline = ML::Metal::PipelineCache.get_or_compile(
    "hy_nomas_query_fixed_phase_gather_f32", METAL_SOURCE)
  roundtrip_pipeline = ML::Metal::PipelineCache.get_or_compile(
    "hy_nomas_query_fixed_phase_bf16_roundtrip_inplace_f32", METAL_SOURCE)
  score_pipeline = ML::Metal::PipelineCache.get_or_compile(
    "hy_nomas_query_correspondence_scores_f32", METAL_SOURCE)
  metric_pipeline = ML::Metal::PipelineCache.get_or_compile(
    "hy_nomas_query_correspondence_metrics_f32", METAL_SOURCE)

  chain = ML::Metal::MPSCommandChain.new
  allocations_before = ML::MetalBuffer.hy_text_probe_allocations
  readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
  waits_before = chain.not_nil!.wait_count
  dispatch_count = 0

  encode_started = Time.instant
  encoder = chain.not_nil!.encoder

  # identity: pre/post roll 0; moving: RoPE then content+phase roll 17;
  # fixed: content roll 17 before destination RoPE, with phases held in place.
  encoder.set_pipeline(gather_pipeline)
    .set_buffer(key, 0).set_buffer(work_key, 1, ML::Metal::BufferAccess::Write)
    .set_value(fixture.pre_key_roll.to_u32, 2)
    .dispatch_1d(TENSOR_VALUES)
  dispatch_count += 1

  encoder.set_pipeline(rope_pipeline)
    .set_buffer(query, 0).set_buffer(cosine, 1).set_buffer(sine, 2)
    .set_buffer(output_query, 3, ML::Metal::BufferAccess::Write)
    .set_value(POSITIONS.to_u32, 4).set_value(HEADS.to_u32, 5)
    .set_value(HEAD_DIM.to_u32, 6).set_value(1_u32, 7)
    .dispatch_1d(TENSOR_VALUES)
  dispatch_count += 1

  encoder.set_pipeline(roundtrip_pipeline)
    .set_buffer(output_query, 0, ML::Metal::BufferAccess::ReadWrite)
    .set_value(TENSOR_VALUES.to_u32, 1).dispatch_1d(TENSOR_VALUES)
  dispatch_count += 1

  encoder.set_pipeline(rope_pipeline)
    .set_buffer(work_key, 0).set_buffer(cosine, 1).set_buffer(sine, 2)
    .set_buffer(work_rope_key, 3, ML::Metal::BufferAccess::Write)
    .set_value(POSITIONS.to_u32, 4).set_value(HEADS.to_u32, 5)
    .set_value(HEAD_DIM.to_u32, 6).set_value(1_u32, 7)
    .dispatch_1d(TENSOR_VALUES)
  dispatch_count += 1

  encoder.set_pipeline(roundtrip_pipeline)
    .set_buffer(work_rope_key, 0, ML::Metal::BufferAccess::ReadWrite)
    .set_value(TENSOR_VALUES.to_u32, 1).dispatch_1d(TENSOR_VALUES)
  dispatch_count += 1

  encoder.set_pipeline(gather_pipeline)
    .set_buffer(work_rope_key, 0).set_buffer(output_key, 1, ML::Metal::BufferAccess::Write)
    .set_value(fixture.post_key_roll.to_u32, 2)
    .dispatch_1d(TENSOR_VALUES)
  dispatch_count += 1

  encoder.set_pipeline(score_pipeline).set_buffer(output_query, 0).set_buffer(output_key, 1)
    .set_buffer(scores, 2, ML::Metal::BufferAccess::Write)
    .dispatch_1d(SCORE_VALUES)
  dispatch_count += 1

  encoder.set_pipeline(metric_pipeline).set_buffer(scores, 0)
    .set_buffer(metrics, 1, ML::Metal::BufferAccess::Write)
    .dispatch_1d(METRIC_COUNT)
  dispatch_count += 1

  encode_seconds = (Time.instant - encode_started).total_seconds
  named_encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
  encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  encode_waits = chain.not_nil!.wait_count - waits_before
  unless dispatch_count == DISPATCH_COUNT && named_encode_allocations == 0 &&
         encode_readbacks == 0 && encode_waits == 0
    raise ArgumentError.new("fixed-phase encode violated its dispatch, allocation, readback, or wait contract")
  end

  wait_started = Time.instant
  chain.not_nil!.commit_and_wait
  wait_seconds = (Time.instant - wait_started).total_seconds
  terminal_waits = chain.not_nil!.wait_count
  raise ArgumentError.new("fixed-phase probe requires exactly one terminal wait") unless terminal_waits == 1

  query_values = output_query.read(TENSOR_VALUES)
  key_values = output_key.read(TENSOR_VALUES)
  score_values = scores.read(SCORE_VALUES)
  metric_values = metrics.read(METRIC_COUNT * METRIC_WIDTH)
  terminal_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
  unless terminal_readbacks == 4
    raise ArgumentError.new("fixed-phase probe requires four post-wait diagnostic readbacks")
  end
  validate_expanded_bf16_values!(query_values, "query")
  validate_expanded_bf16_values!(key_values, "key")
  validate_finite_values!(score_values, "scores")
  validate_finite_values!(metric_values, "metrics")

  query_bytes = float_bytes(query_values)
  output_key_bytes = float_bytes(key_values)
  score_bytes = float_bytes(score_values)
  metric_bytes = float_bytes(metric_values)
  query_sha256 = Digest::SHA256.hexdigest(query_bytes)
  output_key_sha256 = Digest::SHA256.hexdigest(output_key_bytes)
  score_sha256 = Digest::SHA256.hexdigest(score_bytes)
  metric_sha256 = Digest::SHA256.hexdigest(metric_bytes)
  payload_sha = Digest::SHA256.new
  {query_bytes, output_key_bytes, score_bytes, metric_bytes}.each { |bytes| payload_sha.update(bytes) }
  output_payload_sha256 = payload_sha.hexfinal
  output_payload_bytes = query_bytes.size.to_i64 + output_key_bytes.size.to_i64 +
                         score_bytes.size.to_i64 + metric_bytes.size.to_i64
  unless output_payload_bytes == OUTPUT_BYTES
    raise ArgumentError.new("fixed-phase output payload byte count mismatch")
  end

  gpu_elapsed_seconds = chain.not_nil!.gpu_elapsed_seconds
  chain.not_nil!.release
  chain = nil
  arena.not_nil!.release
  arena = nil

  HyNomasQueryCorrespondenceProbe.write_fresh_parts(payload_path,
    [query_bytes, output_key_bytes, score_bytes, metric_bytes])
  report_json = JSON.build(indent: "  ") do |json|
    json.object do
      json.field "schema", OUTPUT_SCHEMA
      json.field "schema_version", 1
      json.field "input_manifest_path", fixture.manifest_path
      json.field "input_manifest_sha256", fixture.manifest_sha256
      json.field "input_qk_manifest_sha256", fixture.qk_manifest_sha256
      json.field "input_phase_payload_sha256", fixture.phase_payload_sha256
      json.field "case" do
        json.object do
          json.field "frame", fixture.frame
          json.field "arm", fixture.arm
          json.field "key_roll", fixture.key_roll
        end
      end
      json.field "geometry", {heads: HEADS, positions: POSITIONS, head_dim: HEAD_DIM,
                              layout: "[positions,heads,head_dim]", score_scale: "1/sqrt(128)"}
      json.field "native_sources" do
        json.object do
          json.field "production_uncontracted_rope_sha256", Digest::SHA256.hexdigest(PRODUCTION_ROPE_SOURCE)
          json.field "production_bf16_roundtrip_helper_sha256", Digest::SHA256.hexdigest(PRODUCTION_BF16_SOURCE)
          json.field "query_correspondence_kernels_sha256", Digest::SHA256.hexdigest(
            HyNomasQueryCorrespondenceProbe::METAL_SOURCE)
          json.field "compiled_metal_source_sha256", Digest::SHA256.hexdigest(METAL_SOURCE)
          json.field "rope_kernel", ML::Video::HyNomasMPSPointwise::ROPE_KERNEL
          json.field "bf16_roundtrip_helper", "hy_nomas_bf16_roundtrip_f32"
        end
      end
      json.field "input_tensors" do
        json.object do
          {"query" => fixture.qk.query_sha256, "key" => fixture.qk.key_sha256}.each do |name, digest|
            json.field name do
              json.object do
                json.field "dtype", "F32"
                json.field "source_dtype", "BF16"
                json.field "shape", [POSITIONS, HEADS, HEAD_DIM]
                json.field "sha256", digest
              end
            end
          end
          {"cosine" => fixture.phase_cosine_bytes, "sine" => fixture.phase_sine_bytes}.each do |name, bytes|
            json.field name do
              json.object do
                json.field "dtype", "F32"
                json.field "shape", [POSITIONS, HEAD_DIM]
                json.field "sha256", Digest::SHA256.hexdigest(bytes)
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
          offset = 0_i64
          {"query"   => {query_sha256, TENSOR_BYTES, [POSITIONS, HEADS, HEAD_DIM], true},
           "key"     => {output_key_sha256, TENSOR_BYTES, [POSITIONS, HEADS, HEAD_DIM], true},
           "scores"  => {score_sha256, SCORE_BYTES, [HEADS, POSITIONS, POSITIONS], false},
           "metrics" => {metric_sha256, METRIC_BYTES, [HEADS, POSITIONS, METRIC_WIDTH], false}}.each do |name, record|
            digest, nbytes, shape, source_bf16 = record
            json.field name do
              json.object do
                json.field "dtype", "F32"
                json.field "source_dtype", "BF16" if source_bf16
                json.field "byte_order", "little"
                json.field "shape", shape
                json.field "offset_bytes", offset
                json.field "nbytes", nbytes
                json.field "sha256", digest
              end
            end
            offset += nbytes
          end
        end
      end
      json.field "metrics_order", METRICS_ORDER
      json.field "execution" do
        json.object do
          json.field "device", "metal"
          json.field "command_chains", 1
          json.field "named_buffer_count", BUFFER_COUNT
          json.field "named_bytes_after_setup", named_bytes
          json.field "named_buffer_bytes", {
            "input_query" => TENSOR_BYTES, "input_key" => TENSOR_BYTES,
            "cosine" => PHASE_BYTES, "sine" => PHASE_BYTES,
            "work_key" => TENSOR_BYTES, "work_rope_key" => TENSOR_BYTES,
            "output_query" => TENSOR_BYTES, "output_key" => TENSOR_BYTES,
            "scores_f32" => SCORE_BYTES, "metrics_f32" => METRIC_BYTES,
          }
          json.field "pre_key_roll", fixture.pre_key_roll
          json.field "post_key_roll", fixture.post_key_roll
          json.field "dispatches", dispatch_count
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
      json.field "scope", "Native Metal fixed-destination-phase key-content null on one authenticated same-pass T0 state; no semantic correspondence, transport, independent-state, or acting claim"
    end
  end
  report_json += "\n"
  HyNomasQueryCorrespondenceProbe.write_fresh_bytes(report_path, report_json.to_slice)
  puts report_path
  puts "PASS: fixed-phase Q/K and correspondence metrics captured after one terminal wait"
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

def float_bytes(values : Array(Float32)) : Bytes
  Bytes.new(values.to_unsafe.as(UInt8*), values.size * sizeof(Float32))
end

def validate_expanded_bf16_values!(values : Array(Float32), label : String) : Nil
  values.each do |value|
    bits = value.unsafe_as(UInt32)
    unless value.finite? && (bits & 0x0000_ffff_u32) == 0
      raise ArgumentError.new("#{label} output is not a finite lossless BF16 expansion")
    end
  end
end

def validate_finite_values!(values : Array(Float32), label : String) : Nil
  unless values.all?(&.finite?)
    raise ArgumentError.new("#{label} output contains non-finite values")
  end
end
