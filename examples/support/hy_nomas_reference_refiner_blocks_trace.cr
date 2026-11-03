require "digest/sha256"
require "json"
require "../../src/ml/core/buffer"
require "./hy_nomas_reference_cache_probe"

module HyNomasReferenceRefinerBlocksTrace
  extend self

  SCHEMA            = "cogni_ml.hy_nomas_reference_refiner_blocks_trace/v1"
  UPSTREAM_REVISION = HyNomasReferenceCacheProbe::UPSTREAM_REVISION
  SCOPE             = "native T0 text-refiner block checkpoints captured after the complete two-block encode on the cache-probe MPS chain; not full upstream T0 preprocessing parity"
  PROCESSED_TOKENS  = 1280
  HIDDEN_DIM        = 2048
  FFN_DIM           = 8192

  # `block1.input` is a named alias of `block0.output`; the payload stores that
  # range once and the manifest records the consumed-input boundary explicitly.
  TENSOR_ORDER = [
    "block0.input", "c",
    "block0.condition_silu", "block0.adaLN", "block0.gate_msa", "block0.gate_mlp",
    "block0.norm1", "block0.qkv", "block0.q", "block0.k", "block0.v",
    "block0.attention", "block0.proj", "block0.residual1", "block0.norm2",
    "block0.mlp_fc1", "block0.mlp_silu", "block0.mlp_fc2", "block0.output",
    "block1.input",
    "block1.condition_silu", "block1.adaLN", "block1.gate_msa", "block1.gate_mlp",
    "block1.norm1", "block1.qkv", "block1.q", "block1.k", "block1.v",
    "block1.attention", "block1.proj", "block1.residual1", "block1.norm2",
    "block1.mlp_fc1", "block1.mlp_silu", "block1.mlp_fc2", "block1.output",
  ]

  SOURCE_FILES = [
    "examples/hy_nomas_reference_cache_probe.cr",
    "examples/support/hy_nomas_reference_cache_probe.cr",
    "examples/support/hy_nomas_reference_refiner_trace.cr",
    "examples/support/hy_nomas_reference_refiner_blocks_trace.cr",
    "src/ml/video/hy_nomas_text_refiner.cr",
    "src/ml/video/kernels/hy_nomas_text_refiner.metal",
    "src/ml/video/metal_ops.cr",
    "src/ml/video/metal_linear_mps.cr",
    "src/ml/video/kernels/linear_mps.metal",
    "src/ml/video/hy_nomas_mps_activation.cr",
    "src/ml/video/hy_nomas_mps_reductions_bridge.mm",
    "src/ml/video/kernels/common.metal",
    "src/ml/video/hy_nomas_attention_mps.cr",
    "src/ml/video/wan_t5.cr",
    "src/ml/video/kernels/wan_t5.metal",
    "src/ml/video/wan_t5_mps_bridge.mm",
  ]

  # Hash the sources embedded in the compiled probe, not later working-tree
  # contents, so the output identifies the executable that made the capture.
  COMPILED_SOURCE = {
    "examples/hy_nomas_reference_cache_probe.cr"                  => {{ read_file("#{__DIR__}/../hy_nomas_reference_cache_probe.cr") }},
    "examples/support/hy_nomas_reference_cache_probe.cr"          => {{ read_file("#{__DIR__}/hy_nomas_reference_cache_probe.cr") }},
    "examples/support/hy_nomas_reference_refiner_trace.cr"        => {{ read_file("#{__DIR__}/hy_nomas_reference_refiner_trace.cr") }},
    "examples/support/hy_nomas_reference_refiner_blocks_trace.cr" => {{ read_file("#{__DIR__}/hy_nomas_reference_refiner_blocks_trace.cr") }},
    "src/ml/video/hy_nomas_text_refiner.cr"                       => {{ read_file("#{__DIR__}/../../src/ml/video/hy_nomas_text_refiner.cr") }},
    "src/ml/video/kernels/hy_nomas_text_refiner.metal"            => {{ read_file("#{__DIR__}/../../src/ml/video/kernels/hy_nomas_text_refiner.metal") }},
    "src/ml/video/metal_ops.cr"                                   => {{ read_file("#{__DIR__}/../../src/ml/video/metal_ops.cr") }},
    "src/ml/video/metal_linear_mps.cr"                            => {{ read_file("#{__DIR__}/../../src/ml/video/metal_linear_mps.cr") }},
    "src/ml/video/kernels/linear_mps.metal"                       => {{ read_file("#{__DIR__}/../../src/ml/video/kernels/linear_mps.metal") }},
    "src/ml/video/hy_nomas_mps_activation.cr"                     => {{ read_file("#{__DIR__}/../../src/ml/video/hy_nomas_mps_activation.cr") }},
    "src/ml/video/hy_nomas_mps_reductions_bridge.mm"              => {{ read_file("#{__DIR__}/../../src/ml/video/hy_nomas_mps_reductions_bridge.mm") }},
    "src/ml/video/kernels/common.metal"                           => {{ read_file("#{__DIR__}/../../src/ml/video/kernels/common.metal") }},
    "src/ml/video/hy_nomas_attention_mps.cr"                      => {{ read_file("#{__DIR__}/../../src/ml/video/hy_nomas_attention_mps.cr") }},
    "src/ml/video/wan_t5.cr"                                      => {{ read_file("#{__DIR__}/../../src/ml/video/wan_t5.cr") }},
    "src/ml/video/kernels/wan_t5.metal"                           => {{ read_file("#{__DIR__}/../../src/ml/video/kernels/wan_t5.metal") }},
    "src/ml/video/wan_t5_mps_bridge.mm"                           => {{ read_file("#{__DIR__}/../../src/ml/video/wan_t5_mps_bridge.mm") }},
  }

  record TensorSource, name : String, checkpoint : String,
    buffer : ML::MetalBuffer, dtype : String, shape : Array(Int32),
    byte_count : Int32, alias_of : String?

  record TensorRecord, dtype : String, shape : Array(Int32), offset : Int64,
    nbytes : Int64, sha256 : String, checkpoint : String, alias_of : String?

  def valid_sha256?(value : String) : Bool
    /\A[0-9a-f]{64}\z/.matches?(value)
  end

  def tensor_spec(name : String) : Tuple(String, Array(Int32))
    case name
    when "block0.input", "block1.input", "block0.norm1", "block0.q", "block0.k", "block0.v",
         "block0.attention", "block0.proj", "block0.residual1", "block0.norm2",
         "block0.mlp_fc2", "block0.output", "block1.norm1", "block1.q", "block1.k",
         "block1.v", "block1.attention", "block1.proj", "block1.residual1", "block1.norm2",
         "block1.mlp_fc2", "block1.output"
      {"F32", [1, PROCESSED_TOKENS, HIDDEN_DIM]}
    when "c", "block0.condition_silu", "block0.gate_msa", "block0.gate_mlp",
         "block1.condition_silu", "block1.gate_msa", "block1.gate_mlp"
      {"F32", [1, HIDDEN_DIM]}
    when "block0.adaLN", "block1.adaLN"
      {"F32", [1, 2 * HIDDEN_DIM]}
    when "block0.qkv", "block1.qkv"
      {"F32", [1, PROCESSED_TOKENS, 3, HIDDEN_DIM]}
    when "block0.mlp_fc1", "block0.mlp_silu", "block1.mlp_fc1", "block1.mlp_silu"
      {"F32", [1, PROCESSED_TOKENS, FFN_DIM]}
    else
      raise ArgumentError.new("unknown reference refiner block trace tensor #{name}")
    end
  end

  def validate_prelude_controls!(prelude_input_bf16_control : Bool,
                                 prelude_linear_bf16_control : Bool,
                                 prelude_silu_bf16_control : Bool,
                                 prelude_second_linear_bf16_control : Bool,
                                 prelude_add_bf16_control : Bool) : Nil
    unless prelude_input_bf16_control && prelude_linear_bf16_control &&
           prelude_silu_bf16_control && prelude_second_linear_bf16_control &&
           prelude_add_bf16_control
      raise ArgumentError.new("refiner block trace requires all five prelude BF16 controls")
    end
  end

  def checkpoint_binding(name : String) : Tuple(String, String?)
    case name
    when "block0.input"
      {"input_embed", nil}
    when "block1.input"
      {"block0.output", "block0.output"}
    else
      tensor_spec(name)
      {name, nil}
    end
  end

  def tensor_sources(checkpoints : Hash(String, ML::MetalBuffer)) : Array(TensorSource)
    TENSOR_ORDER.map do |name|
      checkpoint, alias_of = checkpoint_binding(name)
      buffer = checkpoints[checkpoint]?
      unless buffer
        raise ArgumentError.new("missing reference refiner block checkpoint: #{checkpoint}")
      end
      dtype, shape = tensor_spec(name)
      elements = shape.reduce(1_i64) { |count, extent| count * extent }
      byte_count = (elements * sizeof(Float32)).to_i32
      if alias_name = alias_of
        alias_buffer = checkpoints[alias_name]?
        unless alias_buffer && buffer.same?(alias_buffer)
          raise ArgumentError.new("reference refiner block input alias differs: #{name}")
        end
      end
      TensorSource.new(name, checkpoint, buffer, dtype, shape, byte_count, alias_of)
    end
  end

  def checked_output_paths!(manifest_path : String) : Tuple(String, String)
    unless Path[manifest_path].absolute? && manifest_path.ends_with?(".json")
      raise ArgumentError.new("reference refiner block trace output must be an absolute .json path")
    end
    parent = File.realpath(File.dirname(manifest_path))
    canonical = File.join(parent, File.basename(manifest_path))
    {canonical, canonical.sub(/\.json\z/, ".tensors.bin")}
  end

  def ensure_fresh_outputs!(manifest_path : String, payload_path : String) : Nil
    HyNomasReferenceCacheProbe.ensure_fresh_outputs!(manifest_path, payload_path)
  end

  def ensure_distinct_outputs!(first_manifest : String, first_payload : String,
                               second_manifest : String, second_payload : String) : Nil
    ensure_distinct_paths!([{first_manifest, first_payload}, {second_manifest, second_payload}])
  end

  def ensure_distinct_outputs!(first_manifest : String, first_payload : String,
                               second_manifest : String, second_payload : String,
                               third_manifest : String, third_payload : String) : Nil
    ensure_distinct_paths!([
      {first_manifest, first_payload}, {second_manifest, second_payload},
      {third_manifest, third_payload},
    ])
  end

  private def ensure_distinct_paths!(outputs : Array(Tuple(String, String))) : Nil
    paths = outputs.flat_map(&.to_a)
    unless paths.uniq.size == paths.size
      raise ArgumentError.new("reference cache, prelude trace, and block trace output paths must be distinct")
    end
  end

  def validate_execution!(encode_allocations : Int64, encode_readbacks : Int64,
                          preterminal_waits : Int32, terminal_waits : Int32) : Nil
    unless encode_allocations == 0 && encode_readbacks == 0 &&
           preterminal_waits == 0 && terminal_waits == 1
      raise ArgumentError.new("reference refiner block trace requires allocation/readback-free encode and one terminal drain")
    end
  end

  def compiled_source_sha256s : Hash(String, String)
    SOURCE_FILES.to_h do |relative|
      {relative, Digest::SHA256.hexdigest(COMPILED_SOURCE[relative])}
    end
  end

  def bytes_for(source : TensorSource) : Bytes
    unless source.buffer.valid? && source.buffer.storage_mode.shared? &&
           source.buffer.size >= source.byte_count
      raise ArgumentError.new("reference refiner block capture buffer extent/storage mismatch: #{source.name}")
    end
    bytes = Bytes.new(source.byte_count)
    source.buffer.read_bytes(bytes.to_unsafe, bytes.size)
    bytes
  end

  # Called only after the producer's single terminal wait. All captured bytes
  # come from preallocated checkpoints and are read outside the encode region.
  def validate_qkv_control!(block_boundaries_bf16_control : Bool, qkv_bf16_control : Bool) : Nil
    if qkv_bf16_control && !block_boundaries_bf16_control
      raise ArgumentError.new("QKV BF16 control requires the block-boundary BF16 control")
    end
  end

  def write_trace!(manifest_path : String, *,
                   reference_latent_artifact_sha256 : String,
                   timeline_manifest_sha256 : String,
                   cache_fixture_manifest_sha256 : String,
                   prelude_trace_manifest_sha256 : String,
                   valid_text_tokens : Int32,
                   sources : Array(TensorSource),
                   encode_allocations : Int64,
                   encode_readbacks : Int64,
                   preterminal_waits : Int32,
                   terminal_waits : Int32,
                   prelude_input_bf16_control : Bool,
                   prelude_linear_bf16_control : Bool,
                   prelude_silu_bf16_control : Bool,
                   prelude_second_linear_bf16_control : Bool,
                   prelude_add_bf16_control : Bool,
                   block_boundaries_bf16_control : Bool = false,
                   qkv_bf16_control : Bool = false) : Hash(String, JSON::Any)
    unless valid_text_tokens > 0 && valid_text_tokens <= PROCESSED_TOKENS &&
           valid_sha256?(reference_latent_artifact_sha256) &&
           valid_sha256?(timeline_manifest_sha256) &&
           valid_sha256?(cache_fixture_manifest_sha256) &&
           valid_sha256?(prelude_trace_manifest_sha256)
      raise ArgumentError.new("reference refiner block trace identities or valid-token count are invalid")
    end
    validate_execution!(encode_allocations, encode_readbacks, preterminal_waits, terminal_waits)
    validate_prelude_controls!(prelude_input_bf16_control, prelude_linear_bf16_control,
      prelude_silu_bf16_control, prelude_second_linear_bf16_control, prelude_add_bf16_control)
    validate_qkv_control!(block_boundaries_bf16_control, qkv_bf16_control)
    unless sources.map(&.name) == TENSOR_ORDER
      raise ArgumentError.new("reference refiner block trace source inventory or order mismatch")
    end

    source_by_name = sources.to_h { |source| {source.name, source} }
    canonical, payload_path = checked_output_paths!(manifest_path)
    ensure_fresh_outputs!(canonical, payload_path)
    pins = compiled_source_sha256s
    records = {} of String => TensorRecord
    payload_digest = Digest::SHA256.new
    offset = 0_i64
    payload_io = HyNomasReferenceCacheProbe.open_fresh_output!(payload_path)
    begin
      sources.each do |source|
        expected_dtype, expected_shape = tensor_spec(source.name)
        unless source.dtype == expected_dtype && source.shape == expected_shape
          raise ArgumentError.new("reference refiner block trace source shape/dtype mismatch: #{source.name}")
        end
        expected_bytes = expected_shape.reduce(1_i64) { |count, extent| count * extent } * sizeof(Float32)
        unless source.byte_count.to_i64 == expected_bytes
          raise ArgumentError.new("reference refiner block trace byte count mismatch: #{source.name}")
        end

        if alias_name = source.alias_of
          target_source = source_by_name[alias_name]?
          target_record = records[alias_name]?
          unless target_source && target_record && source.buffer.same?(target_source.buffer) &&
                 source.dtype == target_source.dtype && source.shape == target_source.shape
            raise ArgumentError.new("reference refiner block trace alias target differs: #{source.name}")
          end
          records[source.name] = TensorRecord.new(source.dtype, source.shape,
            target_record.offset, target_record.nbytes, target_record.sha256,
            source.checkpoint, alias_name)
          next
        end

        bytes = bytes_for(source)
        digest = Digest::SHA256.hexdigest(bytes)
        payload_io.write(bytes)
        payload_digest.update(bytes)
        records[source.name] = TensorRecord.new(expected_dtype, expected_shape,
          offset, expected_bytes, digest, source.checkpoint, nil)
        offset += expected_bytes
      end
    ensure
      payload_io.close
    end

    payload_sha256 = payload_digest.hexfinal
    raw = JSON.build(indent: "  ") do |json|
      json.object do
        json.field("schema", SCHEMA)
        json.field("schema_version", 1)
        json.field("upstream_revision", UPSTREAM_REVISION)
        json.field("scope", SCOPE)
        json.field("full_upstream_t0_preprocessing_parity", false)
        json.field("full_text_refiner_encode_executed", true)
        json.field("capture_boundary", "after_block1_output")
        json.field("blocks_executed", 2)
        json.field("processed_tokens", PROCESSED_TOKENS)
        json.field("valid_text_tokens", valid_text_tokens)
        json.field("padding_tokens", PROCESSED_TOKENS - valid_text_tokens)
        json.field("prelude_input_bf16_control", prelude_input_bf16_control)
        json.field("prelude_linear_bf16_control", prelude_linear_bf16_control)
        json.field("prelude_silu_bf16_control", prelude_silu_bf16_control)
        json.field("prelude_second_linear_bf16_control", prelude_second_linear_bf16_control)
        json.field("prelude_add_bf16_control", prelude_add_bf16_control)
        json.field("block_boundaries_bf16_control", block_boundaries_bf16_control)
        json.field("qkv_bf16_control", qkv_bf16_control)
        json.field("device", "mps")
        json.field("producer_dtype", "F32")
        json.field("reference_latent_artifact_sha256", reference_latent_artifact_sha256)
        json.field("timeline_manifest_sha256", timeline_manifest_sha256)
        json.field("cache_fixture_manifest_sha256", cache_fixture_manifest_sha256)
        json.field("prelude_trace_manifest_sha256", prelude_trace_manifest_sha256)
        json.field("compiled_source_sha256") do
          json.object { pins.each { |name, digest| json.field(name, digest) } }
        end
        json.field("tensor_order", TENSOR_ORDER)
        json.field("tensor_artifact") do
          json.object do
            json.field("basename", File.basename(payload_path))
            json.field("bytes", offset)
            json.field("sha256", payload_sha256)
          end
        end
        json.field("tensors") do
          json.object do
            TENSOR_ORDER.each do |name|
              record = records[name]
              json.field(name) do
                json.object do
                  json.field("dtype", record.dtype)
                  json.field("shape", record.shape)
                  json.field("offset", record.offset)
                  json.field("nbytes", record.nbytes)
                  json.field("sha256", record.sha256)
                  json.field("source_checkpoint", record.checkpoint)
                  if alias_name = record.alias_of
                    json.field("alias_of", alias_name)
                  end
                end
              end
            end
          end
        end
        json.field("encode_allocations", encode_allocations)
        json.field("encode_readbacks", encode_readbacks)
        json.field("preterminal_waits", preterminal_waits)
        json.field("terminal_waits", terminal_waits)
      end
    end
    manifest_io = HyNomasReferenceCacheProbe.open_fresh_output!(canonical)
    begin
      manifest_io.puts(raw)
    ensure
      manifest_io.close
    end

    {"manifest"        => JSON::Any.new(canonical),
     "manifest_sha256" => JSON::Any.new(Digest::SHA256.hexdigest(raw + "\n")),
     "payload"         => JSON::Any.new(payload_path),
     "payload_bytes"   => JSON::Any.new(offset),
     "payload_sha256"  => JSON::Any.new(payload_sha256)}
  end
end
