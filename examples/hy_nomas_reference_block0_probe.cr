require "digest/sha256"
require "json"
require "option_parser"
require "../src/ml/video/bf16_weight_cache"
require "../src/ml/video/hy_nomas_block_mps"
require "./support/hy_nomas_reference_block0_probe"
require "./support/hy_nomas_reference_cache_probe"
require "./support/hy_nomas_probe_instrumentation"

module HyNomasReferenceBlock0ProbeCLI
  extend self

  MAX_NAMED_BYTES = HyNomasReferenceBlock0Probe::MAX_NAMED_BYTES
  TEXT_TOKENS     = HyNomasReferenceBlock0Probe::TEXT_STORAGE_TOKENS
  VIDEO_TOKENS    = HyNomasReferenceBlock0Probe::IMAGE_TOKENS
  DIM             = HyNomasReferenceBlock0Probe::DIM
  HEADS           = HyNomasReferenceBlock0Probe::HEADS
  HEAD_DIM        = HyNomasReferenceBlock0Probe::HEAD_DIM
  FFN_DIM         = 4 * DIM

  record CaptureRecord, name : String, shape : Array(Int32), nbytes : Int32,
    sha256 : String
  record PreQKVCaptureRecord, name : String, dtype : String, shape : Array(Int32),
    nbytes : Int32, sha256 : String

  def config(precision : ML::Video::HyNomasAttentionPrecision) : ML::Video::HyNomasConfig
    ML::Video::HyNomasConfig.new(
      batches: 1,
      video_tokens: VIDEO_TOKENS,
      text_tokens: TEXT_TOKENS,
      dim: DIM,
      heads: HEADS,
      head_dim: HEAD_DIM,
      ffn_dim: FFN_DIM,
      attention_precision: precision,
    ).tap(&.validate!)
  end

  def bindings : Array(ML::Video::WeightBinding)
    ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(config(
      ML::Video::HyNomasAttentionPrecision::F32Control),
      ML::GGUF::SafeTensorDType::BF16)
  end

  def input_bytes(info : HyNomasReferenceBlock0Probe::ManifestInfo) : Int64
    named_bytes = 0_i64
    HyNomasReferenceBlock0Probe::INPUT_PREFIX_TENSORS.each do |name|
      if name == "block.txt_valid_f32"
        named_bytes += TEXT_TOKENS.to_i64 * DIM * sizeof(Float32)
      else
        named_bytes += info.tensors[name].nbytes
      end
    end
    named_bytes
  end

  def rms_named_bytes(config : ML::Video::HyNomasConfig) : Int64
    per_stream = ->(tokens : Int32) do
      rows = config.batches.to_i64 * tokens * config.heads
      square = rows * config.head_dim * sizeof(Float32)
      mean = rows * sizeof(Float32)
      2_i64 * (square + mean) # Q and K each own square and row-mean buffers.
    end
    per_stream.call(config.video_tokens) + per_stream.call(config.text_tokens)
  end

  def expected_named_bytes(info : HyNomasReferenceBlock0Probe::ManifestInfo,
                           weight_plan : ML::Video::WeightPlan,
                           config : ML::Video::HyNomasConfig) : Hash(String, Int64)
    scratch = ML::Video::HyNomasDualStreamBlock.scratch_plan(config).sum(0_i64) do |(_name, count)|
      count.to_i64 * sizeof(Float32)
    end
    attention_tokens = VIDEO_TOKENS + info.valid_text_tokens
    attention = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(
      1, attention_tokens, HEADS, HEAD_DIM, 256, HEADS).resident_bytes
    weight_stage = ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(config).to_i64 * sizeof(Float32)
    rms = rms_named_bytes(config)
    inputs = input_bytes(info)
    parts = {
      "input_buffers"                  => inputs,
      "block_weight_residents"         => weight_plan.destination_bytes,
      "block_scratch"                  => scratch,
      "mps_weight_stage"               => weight_stage,
      "compact_attention_buffers"      => attention,
      "qk_rms_square_and_mean_buffers" => rms,
    }
    total = parts.values.sum
    parts["total_named_bytes"] = total
    parts
  end

  def source_buffer(block : ML::Video::HyNomasDualStreamBlock, name : String) : ML::MetalBuffer
    checkpoint_name = case name
                      when "img.q_projection"   then "img.attn.q"
                      when "img.k_projection"   then "img.attn.k"
                      when "img.v_projection"   then "img.attn.v"
                      when "img.q_qknorm"       then "img.attn.q_norm"
                      when "img.k_qknorm"       then "img.attn.k_norm"
                      when "img.q_rope"         then "img.attn.q_rope"
                      when "img.k_rope"         then "img.attn.k_rope"
                      when "txt.q_projection"   then "txt.attn.q"
                      when "txt.k_projection"   then "txt.attn.k"
                      when "txt.v_projection"   then "txt.attn.v"
                      when "txt.q_qknorm"       then "txt.attn.q_norm"
                      when "txt.k_qknorm"       then "txt.attn.k_norm"
                      when "cache.k_native_f32" then "joint.key"
                      when "cache.v_native_f32" then "joint.value"
                      else                           raise ArgumentError.new("unknown reference block capture tensor #{name}")
                      end
    block.checkpoints[checkpoint_name]
  end

  def capture_sources(block : ML::Video::HyNomasDualStreamBlock,
                      valid_text_tokens : Int32) : Array(HyNomasReferenceBlock0Probe::CaptureTensor)
    HyNomasReferenceBlock0Probe::SOURCE_CAPTURE_NAMES.map do |name|
      shape = HyNomasReferenceBlock0Probe.capture_shape(name, valid_text_tokens)
      byte_count = HyNomasReferenceBlock0Probe.capture_byte_count(name, valid_text_tokens)
      HyNomasReferenceBlock0Probe::CaptureTensor.new(name,
        source_buffer(block, name), shape, byte_count,
        name.starts_with?("cache.") ? "prospective_joint_append_value" : "native_pre_attention_block_boundary")
    end
  end

  def preqkv_buffer(block : ML::Video::HyNomasDualStreamBlock, name : String,
                    activation_buffers : Hash(String, ML::MetalBuffer)) : ML::MetalBuffer
    case name
    when "img.mod_silu"   then activation_buffers["img.silu"]
    when "txt.mod_silu"   then activation_buffers["txt.silu"]
    when "img.mod_params" then block.checkpoints["img.modulation"]
    when "txt.mod_params" then block.checkpoints["txt.modulation"]
    when "img.norm1"      then block.checkpoints["img.attn.norm"]
    when "txt.norm1"      then block.checkpoints["txt.attn.norm"]
    when "img.qkv_input"  then block.checkpoints["img.attn.modulated"]
    when "txt.qkv_input"  then block.checkpoints["txt.attn.modulated"]
    else
      raise ArgumentError.new("unknown reference pre-QKV capture tensor #{name}")
    end
  end

  def capture_preqkv_sources(block : ML::Video::HyNomasDualStreamBlock,
                             vec : ML::MetalBuffer, vec_txt : ML::MetalBuffer,
                             valid_text_tokens : Int32) : Array(HyNomasReferenceBlock0Probe::CaptureTensor)
    activation_buffers = {} of String => ML::MetalBuffer
    block.mps_activation_operations(vec, vec_txt).each do |operation|
      activation_buffers[operation.name] = operation.output
    end
    HyNomasReferenceBlock0Probe::PREQKV_CAPTURE_NAMES.map do |name|
      shape = HyNomasReferenceBlock0Probe.preqkv_capture_shape(name, valid_text_tokens)
      byte_count = HyNomasReferenceBlock0Probe.preqkv_capture_byte_count(name, valid_text_tokens)
      HyNomasReferenceBlock0Probe::CaptureTensor.new(name,
        preqkv_buffer(block, name, activation_buffers), shape, byte_count,
        "native_pre_qkv_boundary")
    end
  end

  def output_paths!(artifact_path : String, receipt_path : String) : Tuple(String, String)
    unless Path[artifact_path].absolute? && artifact_path.ends_with?(".safetensors") &&
           Path[receipt_path].absolute? && receipt_path.ends_with?(".json")
      raise ArgumentError.new("reference block outputs must be absolute safetensors and JSON paths")
    end
    artifact = File.join(File.realpath(File.dirname(artifact_path)), File.basename(artifact_path))
    receipt = File.join(File.realpath(File.dirname(receipt_path)), File.basename(receipt_path))
    raise ArgumentError.new("reference block artifact and receipt paths must differ") if artifact == receipt
    ensure_fresh_outputs!(artifact, receipt)
    {artifact, receipt}
  end

  def output_paths_with_preqkv!(artifact_path : String, receipt_path : String,
                                preqkv_artifact_path : String,
                                preqkv_receipt_path : String) : Tuple(String, String, String, String)
    raw_paths = [artifact_path, receipt_path, preqkv_artifact_path, preqkv_receipt_path]
    expected_extensions = [".safetensors", ".json", ".safetensors", ".json"]
    raw_paths.each_with_index do |path, index|
      unless Path[path].absolute? && path.ends_with?(expected_extensions[index])
        raise ArgumentError.new("reference block outputs must be absolute safetensors/JSON paths")
      end
    end
    resolved = raw_paths.map do |path|
      File.join(File.realpath(File.dirname(path)), File.basename(path))
    end
    unless resolved.uniq.size == resolved.size
      raise ArgumentError.new("reference block main and pre-QKV output paths must all differ")
    end
    resolved.each { |path| ensure_fresh_outputs!(path) }
    {resolved[0], resolved[1], resolved[2], resolved[3]}
  end

  def ensure_fresh_outputs!(*paths : String) : Nil
    paths.each do |path|
      if File.exists?(path) || File.symlink?(path)
        raise ArgumentError.new("refusing to replace reference block output: #{path}")
      end
    end
  end

  def open_fresh_output!(path : String) : IO::FileDescriptor
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("reference block output must be fresh and writable: #{path}") if fd < 0
    IO::FileDescriptor.new(fd)
  end

  def write_safetensors!(artifact_path : String,
                         sources : Array(HyNomasReferenceBlock0Probe::CaptureTensor),
                         valid_text_tokens : Int32) : Tuple(String, Array(CaptureRecord), Int64)
    unless sources.map(&.name) == HyNomasReferenceBlock0Probe::SOURCE_CAPTURE_NAMES
      raise ArgumentError.new("reference block capture inventory/order mismatch")
    end
    total_bytes = sources.sum(0_i64) do |source|
      expected_shape = HyNomasReferenceBlock0Probe.capture_shape(source.name, valid_text_tokens)
      expected_bytes = HyNomasReferenceBlock0Probe.capture_byte_count(source.name, valid_text_tokens)
      unless source.byte_count == expected_bytes && source.shape == expected_shape
        raise ArgumentError.new("reference block capture shape/extent mismatch: #{source.name}")
      end
      expected_bytes.to_i64
    end
    entries = HyNomasReferenceBlock0Probe.capture_header_entries(valid_text_tokens)
    header = HyNomasReferenceBlock0Probe.capture_header(entries)
    header_length = Bytes.new(sizeof(UInt64))
    IO::ByteFormat::LittleEndian.encode(header.bytesize.to_u64, header_length)
    total_digest = Digest::SHA256.new
    total_digest.update(header_length)
    total_digest.update(header.to_slice)
    records = [] of CaptureRecord
    offset = 0_i64
    payload_io = open_fresh_output!(artifact_path)
    begin
      payload_io.write(header_length)
      payload_io.write(header.to_slice)
      max_count = sources.max_of(&.byte_count)
      bytes = Bytes.new(max_count)
      sources.each do |source|
        unless source.buffer.valid? && source.buffer.storage_mode.shared? && source.buffer.size >= source.byte_count
          raise ArgumentError.new("reference capture must use a shared F32 source with exact admitted extent: #{source.name}")
        end
        view = bytes[0, source.byte_count]
        source.buffer.read_bytes(view.to_unsafe, source.byte_count)
        digest = Digest::SHA256.hexdigest(view)
        total_digest.update(view)
        payload_io.write(view)
        records << CaptureRecord.new(source.name, source.shape, source.byte_count, digest)
        offset += source.byte_count
      end
      payload_io.flush
      payload_io.fsync
    ensure
      payload_io.close
    end
    unless offset == total_bytes
      raise "reference block tensor payload size mismatch: wrote #{offset}, expected #{total_bytes}"
    end
    expected_file_bytes = HyNomasReferenceBlock0Probe.safetensors_file_bytes(header, total_bytes)
    actual_file_bytes = File.size(artifact_path)
    unless actual_file_bytes == expected_file_bytes
      raise "reference block safetensors file size mismatch: #{actual_file_bytes} != #{expected_file_bytes}"
    end
    {total_digest.hexfinal, records, actual_file_bytes}
  end

  def write_preqkv_safetensors!(artifact_path : String,
                                sources : Array(HyNomasReferenceBlock0Probe::CaptureTensor),
                                valid_text_tokens : Int32, precision : String,
                                parent_native_receipt_sha256 : String,
                                parent_native_artifact_sha256 : String) : Tuple(String, Array(PreQKVCaptureRecord), Int64)
    unless sources.map(&.name) == HyNomasReferenceBlock0Probe::PREQKV_CAPTURE_NAMES
      raise ArgumentError.new("reference pre-QKV capture inventory/order mismatch")
    end
    total_bytes = sources.sum(0_i64) do |source|
      expected_shape = HyNomasReferenceBlock0Probe.preqkv_capture_shape(source.name, valid_text_tokens)
      expected_bytes = HyNomasReferenceBlock0Probe.preqkv_capture_byte_count(source.name, valid_text_tokens)
      unless source.byte_count == expected_bytes && source.shape == expected_shape
        raise ArgumentError.new("reference pre-QKV capture shape/extent mismatch: #{source.name}")
      end
      expected_bytes.to_i64
    end
    entries = HyNomasReferenceBlock0Probe.preqkv_header_entries(valid_text_tokens)
    header = HyNomasReferenceBlock0Probe.capture_header(entries,
      HyNomasReferenceBlock0Probe::PREQKV_CAPTURE_NAMES,
      HyNomasReferenceBlock0Probe.preqkv_metadata_fields(precision,
        parent_native_receipt_sha256, parent_native_artifact_sha256))
    header_length = Bytes.new(sizeof(UInt64))
    IO::ByteFormat::LittleEndian.encode(header.bytesize.to_u64, header_length)
    total_digest = Digest::SHA256.new
    total_digest.update(header_length)
    total_digest.update(header.to_slice)
    records = [] of PreQKVCaptureRecord
    offset = 0_i64
    payload_io = open_fresh_output!(artifact_path)
    begin
      payload_io.write(header_length)
      payload_io.write(header.to_slice)
      max_count = sources.max_of(&.byte_count)
      bytes = Bytes.new(max_count)
      sources.each do |source|
        unless source.buffer.valid? && source.buffer.storage_mode.shared? && source.buffer.size >= source.byte_count
          raise ArgumentError.new("reference pre-QKV capture must use a shared F32 source with admitted extent: #{source.name}")
        end
        view = bytes[0, source.byte_count]
        source.buffer.read_bytes(view.to_unsafe, source.byte_count)
        digest = Digest::SHA256.hexdigest(view)
        total_digest.update(view)
        payload_io.write(view)
        records << PreQKVCaptureRecord.new(source.name, "F32", source.shape, source.byte_count, digest)
        offset += source.byte_count
      end
      payload_io.flush
      payload_io.fsync
    ensure
      payload_io.close
    end
    unless offset == total_bytes
      raise "reference pre-QKV payload size mismatch: wrote #{offset}, expected #{total_bytes}"
    end
    expected_file_bytes = HyNomasReferenceBlock0Probe.safetensors_file_bytes(header, total_bytes)
    actual_file_bytes = File.size(artifact_path)
    unless actual_file_bytes == expected_file_bytes
      raise "reference pre-QKV safetensors file size mismatch: #{actual_file_bytes} != #{expected_file_bytes}"
    end
    {total_digest.hexfinal, records, actual_file_bytes}
  end

  def write_receipt!(path : String, *, artifact_basename : String,
                     artifact_sha256 : String, artifact_bytes : Int64,
                     records : Array(CaptureRecord), input : HyNomasReferenceBlock0Probe::InputBundle,
                     weights_sha256 : String, precision : String,
                     valid_text_tokens : Int32, named_bytes : Hash(String, Int64),
                     encode_allocations : Int64, encode_readbacks : Int64,
                     preterminal_waits : Int32, terminal_waits : Int32) : String
    receipt = JSON.build(indent: "  ") do |json|
      json.object do
        json.field("schema", HyNomasReferenceBlock0Probe::SCHEMA)
        json.field("schema_version", 1)
        json.field("native_revision", HyNomasReferenceBlock0Probe::NATIVE_REVISION)
        json.field("source_revision", HyNomasReferenceBlock0Probe::SOURCE_REVISION)
        json.field("scope", "one standalone native block-0 forward using native-exported T0 preblock inputs; not a complete upstream T0 handoff")
        json.field("full_upstream_t0_preprocessing_parity", false)
        json.field("history_cache_write_executed", false)
        json.field("cache_tensor_semantics", "prospective concatenated K/V before any continuation-history append")
        json.field("input_bundle_manifest_sha256", input.info.manifest_sha256)
        json.field("input_bundle_payload_sha256", input.info.payload_sha256)
        json.field("reference_latent_artifact_sha256", input.info.reference_latent_artifact_sha256)
        json.field("timeline_manifest_sha256", input.info.timeline_manifest_sha256)
        json.field("valid_text_tokens", valid_text_tokens)
        json.field("text_storage_tokens", TEXT_TOKENS)
        json.field("text_padding_rows", TEXT_TOKENS - valid_text_tokens)
        json.field("text_padding_value", 0)
        json.field("weights_sha256", weights_sha256)
        json.field("precision_control", precision)
        json.field("tensor_artifact") do
          json.object do
            json.field("basename", artifact_basename)
            json.field("sha256", artifact_sha256)
            json.field("bytes", artifact_bytes)
            json.field("dtype", "F32")
            json.field("tensors") do
              json.array do
                records.each do |record|
                  json.object do
                    json.field("name", record.name)
                    json.field("shape", record.shape)
                    json.field("nbytes", record.nbytes)
                    json.field("sha256", record.sha256)
                  end
                end
              end
            end
          end
        end
        json.field("named_buffer_accounting") do
          json.object do
            named_bytes.each { |name, bytes| json.field(name, bytes) }
            json.field("opaque_mpsgraph_private_bytes", "not measured; excluded from named Metal-buffer total")
          end
        end
        json.field("execution_counters") do
          json.object do
            json.field("encode_allocations", encode_allocations)
            json.field("encode_readbacks", encode_readbacks)
            json.field("preterminal_waits", preterminal_waits)
            json.field("terminal_waits", terminal_waits)
          end
        end
      end
    end
    receipt_bytes = receipt + "\n"
    io = open_fresh_output!(path)
    begin
      io.write(receipt_bytes.to_slice)
      io.flush
      io.fsync
    ensure
      io.close
    end
    Digest::SHA256.hexdigest(receipt_bytes)
  end

  def write_preqkv_receipt!(path : String, *, artifact_basename : String,
                            artifact_sha256 : String, artifact_bytes : Int64,
                            records : Array(PreQKVCaptureRecord),
                            input : HyNomasReferenceBlock0Probe::InputBundle,
                            weights_sha256 : String, precision : String,
                            parent_native_receipt_sha256 : String,
                            parent_native_artifact_sha256 : String) : Nil
    receipt = JSON.build(indent: "  ") do |json|
      json.object do
        json.field("schema", HyNomasReferenceBlock0Probe::PREQKV_SCHEMA)
        json.field("schema_version", 1)
        json.field("native_revision", HyNomasReferenceBlock0Probe::NATIVE_REVISION)
        json.field("source_revision", HyNomasReferenceBlock0Probe::SOURCE_REVISION)
        json.field("scope", HyNomasReferenceBlock0Probe::PREQKV_SCOPE)
        json.field("precision_control", HyNomasReferenceBlock0Probe.preqkv_precision_label(precision))
        json.field("input_bundle_manifest_sha256", input.info.manifest_sha256)
        json.field("input_bundle_payload_sha256", input.info.payload_sha256)
        json.field("reference_latent_artifact_sha256", input.info.reference_latent_artifact_sha256)
        json.field("timeline_manifest_sha256", input.info.timeline_manifest_sha256)
        json.field("valid_text_tokens", input.info.valid_text_tokens)
        json.field("weights_sha256", weights_sha256)
        json.field("parent_native_receipt_sha256", parent_native_receipt_sha256)
        json.field("parent_native_artifact_sha256", parent_native_artifact_sha256)
        json.field("tensor_artifact") do
          json.object do
            json.field("basename", artifact_basename)
            json.field("sha256", artifact_sha256)
            json.field("bytes", artifact_bytes)
            json.field("dtype", "F32")
            json.field("tensors") do
              json.array do
                records.each do |record|
                  json.object do
                    json.field("name", record.name)
                    json.field("dtype", record.dtype)
                    json.field("shape", record.shape)
                    json.field("nbytes", record.nbytes)
                    json.field("sha256", record.sha256)
                  end
                end
              end
            end
          end
        end
      end
    end
    io = open_fresh_output!(path)
    begin
      io.puts(receipt)
      io.flush
      io.fsync
    ensure
      io.close
    end
  end
end

inputs_path = ""
inputs_sha256 = ""
weights_path = ""
weights_sha256 = ""
artifact_path = ""
receipt_path = ""
preqkv_artifact_path = ""
preqkv_receipt_path = ""
precision_name = ""
max_named_bytes = 4_i64 * 1024 * 1024 * 1024
execute = false
capture_preqkv = false
OptionParser.parse do |parser|
  parser.banner = "Standalone Hy-Nomas block-0 reference cache diagnostic."
  parser.on("--inputs PATH", "Authenticated native T0 preblock input manifest") { |value| inputs_path = value }
  parser.on("--inputs-sha256 SHA", "Externally trusted input-manifest SHA-256") { |value| inputs_sha256 = value }
  parser.on("--weights PATH", "Pinned BF16 block-0 checkpoint") { |value| weights_path = value }
  parser.on("--weights-sha256 SHA", "Trusted block-0 checkpoint SHA-256") { |value| weights_sha256 = value }
  parser.on("--output-safetensors PATH", "Fresh F32 tensor artifact; required with --execute") { |value| artifact_path = value }
  parser.on("--output-receipt PATH", "Fresh JSON receipt; required with --execute") { |value| receipt_path = value }
  parser.on("--capture-preqkv", "Also capture native pre-QKV intermediates into a bound companion artifact") { capture_preqkv = true }
  parser.on("--preqkv-output-safetensors PATH", "Fresh pre-QKV companion tensor artifact") { |value| preqkv_artifact_path = value }
  parser.on("--preqkv-output-receipt PATH", "Fresh pre-QKV companion JSON receipt") { |value| preqkv_receipt_path = value }
  parser.on("--precision MODE", "Explicit f32-control or bf16-boundary-control diagnostic") { |value| precision_name = value }
  parser.on("--max-named-bytes N", "Named Metal-buffer ceiling, at most 4 GiB") { |value| max_named_bytes = value.to_i64 }
  parser.on("--execute", "Run one block and capture after the sole terminal drain") { execute = true }
end

unless ARGV.empty? && !inputs_path.empty? && !inputs_sha256.empty? &&
       !weights_path.empty? && !weights_sha256.empty? &&
       {"f32-control", "bf16-boundary-control"}.includes?(precision_name) &&
       max_named_bytes > 0 && max_named_bytes <= HyNomasReferenceBlock0ProbeCLI::MAX_NAMED_BYTES
  raise ArgumentError.new("authenticated inputs, pinned weights, explicit precision and a positive named-byte cap are required")
end
HyNomasReferenceBlock0Probe.validate_preqkv_options!(capture_preqkv, execute,
  preqkv_artifact_path, preqkv_receipt_path)
unless HyNomasReferenceBlock0Probe.valid_sha256?(weights_sha256) &&
       weights_sha256 == HyNomasReferenceBlock0Probe::BLOCK0_WEIGHT_SHA256
  raise ArgumentError.new("block-0 weights must match the pinned BF16 source SHA-256")
end
raise ArgumentError.new("--execute requires fresh safetensors and receipt paths") if execute &&
                                                                                     (artifact_path.empty? || receipt_path.empty?)

input_bundle = HyNomasReferenceBlock0Probe::InputBundle.new(inputs_path, inputs_sha256)
weight_file = nil.as(ML::Video::CheckpointFile?)
weight_plan = nil.as(ML::Video::WeightPlan?)
runtime_owners_retained = false
begin
  unless Path[weights_path].absolute?
    raise ArgumentError.new("block-0 checkpoint path must be absolute")
  end
  weight_file = ML::Video::BF16WeightCache.authenticate(weights_path, weights_sha256)
  precision = precision_name == "bf16-boundary-control" ? ML::Video::HyNomasAttentionPrecision::BF16BoundaryControl : ML::Video::HyNomasAttentionPrecision::F32Control
  config = HyNomasReferenceBlock0ProbeCLI.config(precision)
  weight_plan = ML::Video::HyNomasDualStreamBlock.weight_plan([weight_file.not_nil!], config,
    max_named_bytes, resident_dtype: ML::Video::WeightDType::BF16, require_all: true)
  unless weight_plan.not_nil!.staging_bytes == 0 &&
         weight_plan.not_nil!.entries.size == HyNomasReferenceBlock0ProbeCLI.bindings.size
    raise ArgumentError.new("block-0 weights must be an exact direct BF16 resident plan")
  end

  named_bytes = HyNomasReferenceBlock0ProbeCLI.expected_named_bytes(input_bundle.info,
    weight_plan.not_nil!, config)
  total_named_bytes = named_bytes["total_named_bytes"]
  unless total_named_bytes <= max_named_bytes
    raise ArgumentError.new("reference block named Metal-buffer plan exceeds its ceiling")
  end
  if execute
    if capture_preqkv
      artifact_path, receipt_path, preqkv_artifact_path, preqkv_receipt_path =
        HyNomasReferenceBlock0ProbeCLI.output_paths_with_preqkv!(artifact_path, receipt_path,
          preqkv_artifact_path, preqkv_receipt_path)
    else
      artifact_path, receipt_path = HyNomasReferenceBlock0ProbeCLI.output_paths!(artifact_path, receipt_path)
    end
  end

  plan_document = {
    schema:                                HyNomasReferenceBlock0Probe::SCHEMA,
    scope:                                 "one standalone native block-0 forward; not a complete upstream T0 handoff",
    inputs_manifest_sha256:                input_bundle.info.manifest_sha256,
    inputs_payload_sha256:                 input_bundle.info.payload_sha256,
    reference_latent_artifact_sha256:      input_bundle.info.reference_latent_artifact_sha256,
    timeline_manifest_sha256:              input_bundle.info.timeline_manifest_sha256,
    valid_text_tokens:                     input_bundle.info.valid_text_tokens,
    text_storage_tokens:                   HyNomasReferenceBlock0ProbeCLI::TEXT_TOKENS,
    weights_sha256:                        weights_sha256,
    precision_control:                     precision_name,
    named_buffer_accounting:               named_bytes,
    max_named_bytes:                       max_named_bytes,
    opaque_mpsgraph_private_bytes:         "not measured; requires external memory guard",
    execute:                               execute,
    capture_preqkv:                        capture_preqkv,
    full_upstream_t0_preprocessing_parity: false,
    history_cache_write_executed:          false,
  }
  unless execute
    puts plan_document.to_pretty_json
    exit 0
  end

  input_arena = nil.as(ML::Video::Arena?)
  weight_arena = nil.as(ML::Video::Arena?)
  scratch_arena = nil.as(ML::Video::Arena?)
  prepared = nil.as(ML::Video::PreparedWeights?)
  block = nil.as(ML::Video::HyNomasDualStreamBlock?)
  workspace = nil.as(ML::Video::HyNomasMPSWorkspace?)
  chain = nil.as(ML::Metal::MPSCommandChain?)
  wait_attempted = false

  begin
    input_arena = ML::Video::Arena.new
    weight_arena = ML::Video::Arena.new
    scratch_arena = ML::Video::Arena.new
    img = input_bundle.upload!("block.img_f32", input_arena.not_nil!)
    txt = input_bundle.upload_text_storage!(input_arena.not_nil!)
    vec = input_bundle.upload!("block.vec_f32", input_arena.not_nil!)
    vec_txt = input_bundle.upload!("block.vec_txt_f32", input_arena.not_nil!)
    cosine = input_bundle.upload!("block.rope_cos_f32", input_arena.not_nil!)
    sine = input_bundle.upload!("block.rope_sin_f32", input_arena.not_nil!)

    ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch_arena.not_nil!, config)
    prepared = ML::Video::PreparedWeights.new(weight_plan.not_nil!, weight_arena.not_nil!)
    block = ML::Video::HyNomasDualStreamBlock.new(config, weight_arena.not_nil!, scratch_arena.not_nil!)
    workspace = ML::Video::HyNomasMPSWorkspace.new(block.not_nil!, img, txt, vec, vec_txt,
      cosine, sine, attention_query_chunk: 256,
      attention_heads_per_group: HyNomasReferenceBlock0ProbeCLI::HEADS,
      valid_text_tokens: input_bundle.info.valid_text_tokens,
      text_rms_tokens: HyNomasReferenceBlock0ProbeCLI::TEXT_TOKENS)

    actual_named_bytes = input_arena.not_nil!.resident_bytes + weight_arena.not_nil!.resident_bytes +
                         scratch_arena.not_nil!.resident_bytes + workspace.not_nil!.resident_bytes
    unless actual_named_bytes == total_named_bytes && actual_named_bytes <= max_named_bytes
      raise ArgumentError.new("reference block allocation plan mismatch: #{actual_named_bytes} != #{total_named_bytes}")
    end

    chain = ML::Metal::MPSCommandChain.new
    allocations_before = ML::MetalBuffer.hy_text_probe_allocations
    readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
    block.not_nil!.encode_mps(chain.not_nil!, img, txt, vec, vec_txt, cosine, sine,
      workspace.not_nil!)
    encode_allocations = ML::MetalBuffer.hy_text_probe_allocations - allocations_before
    encode_readbacks = ML::MetalBuffer.hy_text_probe_readbacks - readbacks_before
    preterminal_waits = chain.not_nil!.wait_count
    unless encode_allocations == 0 && encode_readbacks == 0 && preterminal_waits == 0
      raise "block-0 encode allocated, read back, or waited before its terminal drain"
    end

    wait_attempted = true
    chain.not_nil!.commit_and_wait
    unless chain.not_nil!.drained? && chain.not_nil!.wait_count == 1
      raise "block-0 capture requires exactly one completed terminal drain"
    end

    sources = HyNomasReferenceBlock0ProbeCLI.capture_sources(block.not_nil!,
      input_bundle.info.valid_text_tokens)
    artifact_sha256, records, artifact_bytes = HyNomasReferenceBlock0ProbeCLI.write_safetensors!(
      artifact_path, sources, input_bundle.info.valid_text_tokens)
    main_receipt_sha256 = HyNomasReferenceBlock0ProbeCLI.write_receipt!(receipt_path,
      artifact_basename: File.basename(artifact_path), artifact_sha256: artifact_sha256,
      artifact_bytes: artifact_bytes, records: records, input: input_bundle,
      weights_sha256: weights_sha256, precision: precision_name,
      valid_text_tokens: input_bundle.info.valid_text_tokens, named_bytes: named_bytes,
      encode_allocations: encode_allocations, encode_readbacks: encode_readbacks,
      preterminal_waits: preterminal_waits, terminal_waits: chain.not_nil!.wait_count)
    if capture_preqkv
      preqkv_sources = HyNomasReferenceBlock0ProbeCLI.capture_preqkv_sources(block.not_nil!,
        vec, vec_txt, input_bundle.info.valid_text_tokens)
      preqkv_sha256, preqkv_records, preqkv_bytes =
        HyNomasReferenceBlock0ProbeCLI.write_preqkv_safetensors!(preqkv_artifact_path,
          preqkv_sources, input_bundle.info.valid_text_tokens, precision_name,
          main_receipt_sha256, artifact_sha256)
      HyNomasReferenceBlock0ProbeCLI.write_preqkv_receipt!(preqkv_receipt_path,
        artifact_basename: File.basename(preqkv_artifact_path),
        artifact_sha256: preqkv_sha256, artifact_bytes: preqkv_bytes,
        records: preqkv_records, input: input_bundle,
        weights_sha256: weights_sha256, precision: precision_name,
        parent_native_receipt_sha256: main_receipt_sha256,
        parent_native_artifact_sha256: artifact_sha256)
      puts({schema: HyNomasReferenceBlock0Probe::PREQKV_SCHEMA,
            artifact: File.basename(preqkv_artifact_path),
            receipt: File.basename(preqkv_receipt_path),
            artifact_sha256: preqkv_sha256, artifact_bytes: preqkv_bytes,
            parent_native_receipt_sha256: main_receipt_sha256,
            parent_native_artifact_sha256: artifact_sha256,
            passed: true}.to_pretty_json)
    end
    puts plan_document.to_pretty_json
    puts({schema: HyNomasReferenceBlock0Probe::SCHEMA, artifact: File.basename(artifact_path),
          receipt: File.basename(receipt_path), artifact_sha256: artifact_sha256,
          artifact_bytes: artifact_bytes, encode_allocations: encode_allocations,
          encode_readbacks: encode_readbacks, preterminal_waits: preterminal_waits,
          terminal_waits: chain.not_nil!.wait_count,
          history_cache_write_executed: false, passed: true}.to_pretty_json)
  ensure
    if active_chain = chain
      unless wait_attempted || active_chain.drained?
        wait_attempted = true
        begin
          active_chain.commit_and_wait
        rescue
        end
      end
      if active_chain.drained?
        workspace.try(&.release)
        prepared.try(&.release)
        active_chain.release
        scratch_arena.try(&.release)
        input_arena.try(&.release)
        weight_arena.try(&.release)
        input_bundle.close
        weight_file.not_nil!.close
      else
        runtime_owners_retained = true
        HyNomasReferenceCacheProbe.retain_in_flight!(-> {
          active_chain.object_id
          workspace.try(&.object_id)
          block.try(&.object_id)
          prepared.try(&.object_id)
          scratch_arena.try(&.object_id)
          input_arena.try(&.object_id)
          weight_arena.try(&.object_id)
          input_bundle.object_id
          weight_file.try(&.object_id)
          nil
        })
      end
    else
      workspace.try(&.release)
      prepared.try(&.release)
      scratch_arena.try(&.release)
      input_arena.try(&.release)
      weight_arena.try(&.release)
      input_bundle.close
      weight_file.not_nil!.close
    end
  end
ensure
  unless runtime_owners_retained
    input_bundle.close
    weight_file.try(&.close)
  end
end
