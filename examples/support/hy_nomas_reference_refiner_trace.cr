require "digest/sha256"
require "json"
require "../../src/ml/core/buffer"
require "./hy_nomas_reference_cache_probe"

module HyNomasReferenceRefinerTrace
  extend self

  SCHEMA            = "cogni_ml.hy_nomas_reference_refiner_trace/v1"
  UPSTREAM_REVISION = HyNomasReferenceCacheProbe::UPSTREAM_REVISION
  SCOPE             = "native T0 preblock checkpoints captured after the complete text-refiner encode on the cache-probe MPS chain; not full upstream T0 preprocessing parity"
  INPUT_DIM         = 3584
  HIDDEN_DIM        = 2048
  TIMESTEP_DIM      =  256

  # Names and order match the upstream source preblock trace comparator.
  TENSOR_ORDER = [
    "input_compact_valid", "masked_mean", "timestep",
    "refiner_timestep_sinusoidal", "refiner_timestep_mlp1",
    "refiner_timestep_silu", "refiner_timestep", "context_linear1",
    "context_silu", "context_embed", "c", "input_embed",
    "time_in.sinusoidal", "time_in.mlp1", "time_in.silu", "vec",
  ]

  SOURCE_FILES = [
    "examples/hy_nomas_reference_cache_probe.cr",
    "examples/support/hy_nomas_reference_cache_probe.cr",
    "examples/support/hy_nomas_reference_refiner_trace.cr",
    "src/ml/video/hy_nomas_text_refiner.cr",
    "src/ml/metal/kernels/hy_nomas/hy_nomas_text_refiner.metal",
    "src/ml/video/metal_ops.cr",
    "src/ml/video/metal_linear_mps.cr",
    "src/ml/metal/kernels/shared/linear_mps.metal",
    "src/ml/video/hy_nomas_mps_activation.cr",
    "src/ml/video/hy_nomas_mps_reductions_bridge.mm",
    "src/ml/metal/kernels/hy_nomas/common.metal",
    "src/ml/video/wan_t5.cr",
    "src/ml/metal/kernels/shared/wan_t5.metal",
    "src/ml/video/wan_t5_mps_bridge.mm",
  ]

  # These bytes are embedded when Crystal compiles the probe. Hashing the
  # snapshots prevents a later working-tree edit from relabeling an older
  # executable's capture as if it used the newer source.
  COMPILED_SOURCE = {
    "examples/hy_nomas_reference_cache_probe.cr"                => {{ read_file("#{__DIR__}/../hy_nomas_reference_cache_probe.cr") }},
    "examples/support/hy_nomas_reference_cache_probe.cr"        => {{ read_file("#{__DIR__}/hy_nomas_reference_cache_probe.cr") }},
    "examples/support/hy_nomas_reference_refiner_trace.cr"      => {{ read_file("#{__DIR__}/hy_nomas_reference_refiner_trace.cr") }},
    "src/ml/video/hy_nomas_text_refiner.cr"                     => {{ read_file("#{__DIR__}/../../src/ml/video/hy_nomas_text_refiner.cr") }},
    "src/ml/metal/kernels/hy_nomas/hy_nomas_text_refiner.metal" => {{ read_file("#{__DIR__}/../../src/ml/metal/kernels/hy_nomas/hy_nomas_text_refiner.metal") }},
    "src/ml/video/metal_ops.cr"                                 => {{ read_file("#{__DIR__}/../../src/ml/video/metal_ops.cr") }},
    "src/ml/video/metal_linear_mps.cr"                          => {{ read_file("#{__DIR__}/../../src/ml/video/metal_linear_mps.cr") }},
    "src/ml/metal/kernels/shared/linear_mps.metal"              => {{ read_file("#{__DIR__}/../../src/ml/metal/kernels/shared/linear_mps.metal") }},
    "src/ml/video/hy_nomas_mps_activation.cr"                   => {{ read_file("#{__DIR__}/../../src/ml/video/hy_nomas_mps_activation.cr") }},
    "src/ml/video/hy_nomas_mps_reductions_bridge.mm"            => {{ read_file("#{__DIR__}/../../src/ml/video/hy_nomas_mps_reductions_bridge.mm") }},
    "src/ml/metal/kernels/hy_nomas/common.metal"                => {{ read_file("#{__DIR__}/../../src/ml/metal/kernels/hy_nomas/common.metal") }},
    "src/ml/video/wan_t5.cr"                                    => {{ read_file("#{__DIR__}/../../src/ml/video/wan_t5.cr") }},
    "src/ml/metal/kernels/shared/wan_t5.metal"                  => {{ read_file("#{__DIR__}/../../src/ml/metal/kernels/shared/wan_t5.metal") }},
    "src/ml/video/wan_t5_mps_bridge.mm"                         => {{ read_file("#{__DIR__}/../../src/ml/video/wan_t5_mps_bridge.mm") }},
  }

  record TensorSource, name : String, checkpoint : String,
    buffer : ML::MetalBuffer, dtype : String, shape : Array(Int32),
    byte_count : Int32, alias_of : String?

  record TensorRecord, dtype : String, shape : Array(Int32), offset : Int64,
    nbytes : Int64, sha256 : String, checkpoint : String, alias_of : String?

  def valid_sha256?(value : String) : Bool
    /\A[0-9a-f]{64}\z/.matches?(value)
  end

  def tensor_spec(name : String, valid_text_tokens : Int32) : Tuple(String, Array(Int32))
    unless 0 < valid_text_tokens <= 1280
      raise ArgumentError.new("reference refiner trace requires 1..1280 valid text tokens")
    end

    case name
    when "input_compact_valid"                               then {"F32", [1, valid_text_tokens, INPUT_DIM]}
    when "masked_mean"                                       then {"F32", [1, INPUT_DIM]}
    when "timestep"                                          then {"F32", [1]}
    when "refiner_timestep_sinusoidal", "time_in.sinusoidal" then {"F32", [1, TIMESTEP_DIM]}
    when "refiner_timestep_mlp1", "refiner_timestep_silu", "refiner_timestep",
         "context_linear1", "context_silu", "context_embed", "c",
         "time_in.mlp1", "time_in.silu", "vec"
      {"F32", [1, HIDDEN_DIM]}
    when "input_embed" then {"F32", [1, valid_text_tokens, HIDDEN_DIM]}
    else                    raise ArgumentError.new("unknown reference refiner trace tensor #{name}")
    end
  end

  def tensor_sources(checkpoints : Hash(String, ML::MetalBuffer),
                     valid_text_tokens : Int32) : Array(TensorSource)
    # This exact alias is established by HyNomasTextRefinerMPSPlan's buffer
    # allocation and is part of the trace contract.
    unless checkpoints.has_key?("refiner_timestep_sinusoidal") &&
           checkpoints.has_key?("time_in.sinusoidal") &&
           checkpoints["time_in.sinusoidal"].same?(checkpoints["refiner_timestep_sinusoidal"])
      raise ArgumentError.new("reference refiner sinusoidal checkpoint alias differs")
    end

    names = [
      {"input_compact_valid", "input_compact"},
      {"masked_mean", "masked_mean"},
      {"timestep", "timestep"},
      {"refiner_timestep_sinusoidal", "refiner_timestep_sinusoidal"},
      {"refiner_timestep_mlp1", "refiner_timestep_mlp1"},
      {"refiner_timestep_silu", "refiner_timestep_silu"},
      {"refiner_timestep", "refiner_timestep"},
      {"context_linear1", "context_linear1"},
      {"context_silu", "context_silu"},
      {"context_embed", "context_embed"},
      {"c", "c"},
      {"input_embed", "input_embed"},
      {"time_in.sinusoidal", "time_in.sinusoidal"},
      {"time_in.mlp1", "time_in.mlp1"},
      {"time_in.silu", "time_in.silu"},
      {"vec", "vec"},
    ]

    names.map do |name, checkpoint|
      dtype, shape = tensor_spec(name, valid_text_tokens)
      alias_of = name == "time_in.sinusoidal" ? "refiner_timestep_sinusoidal" : nil
      elements = shape.reduce(1_i64) { |count, extent| count * extent }
      byte_count = (elements * sizeof(Float32)).to_i32
      unless checkpoints.has_key?(checkpoint)
        raise ArgumentError.new("missing reference refiner checkpoint: #{checkpoint}")
      end
      TensorSource.new(name, checkpoint, checkpoints[checkpoint], dtype, shape,
        byte_count, alias_of)
    end
  end

  def checked_output_paths!(manifest_path : String) : Tuple(String, String)
    unless Path[manifest_path].absolute? && manifest_path.ends_with?(".json")
      raise ArgumentError.new("reference refiner trace output must be an absolute .json path")
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
    first = {first_manifest, first_payload}
    second = {second_manifest, second_payload}
    unless first.to_a.all? { |left| second.to_a.all? { |right| left != right } }
      raise ArgumentError.new("reference cache and refiner trace output paths must be distinct")
    end
  end

  def validate_execution!(encode_allocations : Int64, encode_readbacks : Int64,
                          preterminal_waits : Int32, terminal_waits : Int32) : Nil
    unless encode_allocations == 0 && encode_readbacks == 0 &&
           preterminal_waits == 0 && terminal_waits == 1
      raise ArgumentError.new("reference refiner trace requires allocation/readback-free encode and one terminal drain")
    end
  end

  def prelude_metadata(prelude_input_bf16_control : Bool = false,
                       prelude_linear_bf16_control : Bool = false,
                       prelude_silu_bf16_control : Bool = false,
                       prelude_second_linear_bf16_control : Bool = false,
                       prelude_add_bf16_control : Bool = false)
    if prelude_linear_bf16_control && !prelude_input_bf16_control
      raise ArgumentError.new("linear BF16 control requires input BF16 control")
    end
    if prelude_silu_bf16_control && !prelude_linear_bf16_control
      raise ArgumentError.new("SiLU BF16 control requires linear BF16 control")
    end
    if prelude_second_linear_bf16_control && !prelude_silu_bf16_control
      raise ArgumentError.new("second-linear BF16 control requires SiLU BF16 control")
    end
    if prelude_add_bf16_control && !prelude_second_linear_bf16_control
      raise ArgumentError.new("add BF16 control requires second-linear BF16 control")
    end
    {
      prelude_input_bf16_control:               prelude_input_bf16_control,
      prelude_linear_bf16_control:              prelude_linear_bf16_control,
      prelude_silu_bf16_control:                prelude_silu_bf16_control,
      prelude_second_linear_bf16_control:       prelude_second_linear_bf16_control,
      prelude_add_bf16_control:                 prelude_add_bf16_control,
      full_bf16_prelude_parity:                 false,
      prelude_input_bf16_control_scope:         prelude_input_bf16_control ? "masked-mean sum, denominator, and quotient plus sinusoidal cos/sin output roundtrip only" : "disabled",
      prelude_linear_bf16_control_scope:        prelude_linear_bf16_control ? "post-bias output roundtrip of refiner_timestep_mlp1, context_linear1, input_embed, and time_in.mlp1 only" : "disabled",
      prelude_silu_bf16_control_scope:          prelude_silu_bf16_control ? "BF16-declared negative, exp, add, and divide with explicit RNE output expansion in the three first-prelude SiLU graphs only; backend internal rounding unqualified" : "disabled",
      prelude_second_linear_bf16_control_scope: prelude_second_linear_bf16_control ? "post-bias RNE-to-BF16 output roundtrip of refiner_timestep, context_embed, and time_in.mlp2 (trace tensor vec) projections only" : "disabled",
      prelude_add_bf16_control_scope:           prelude_add_bf16_control ? "RNE-to-BF16 roundtrip of both operands and the c = refiner_timestep + context_embed sum only" : "disabled",
    }
  end

  # Hash the source snapshots embedded at compile time, not files that may
  # have changed on disk since the running executable was built.
  def compiled_source_sha256s : Hash(String, String)
    SOURCE_FILES.to_h do |relative|
      {relative, Digest::SHA256.hexdigest(COMPILED_SOURCE[relative])}
    end
  end

  def bytes_for(source : TensorSource) : Bytes
    unless source.buffer.valid? && source.buffer.storage_mode.shared? &&
           source.buffer.size >= source.byte_count
      raise ArgumentError.new("reference refiner capture buffer extent/storage mismatch: #{source.name}")
    end
    bytes = Bytes.new(source.byte_count)
    source.buffer.read_bytes(bytes.to_unsafe, bytes.size)
    bytes
  end

  # Called only after the producer's one terminal wait. The source cache fixture
  # manifest hash links this optional trace to the unchanged nine-tensor export.
  def write_trace!(manifest_path : String, *,
                   reference_latent_artifact_sha256 : String,
                   timeline_manifest_sha256 : String,
                   cache_fixture_manifest_sha256 : String,
                   valid_text_tokens : Int32,
                   sources : Array(TensorSource),
                   encode_allocations : Int64,
                   encode_readbacks : Int64,
                   preterminal_waits : Int32,
                   terminal_waits : Int32,
                   prelude_input_bf16_control : Bool = false,
                   prelude_linear_bf16_control : Bool = false,
                   prelude_silu_bf16_control : Bool = false,
                   prelude_second_linear_bf16_control : Bool = false,
                   prelude_add_bf16_control : Bool = false) : Hash(String, JSON::Any)
    unless valid_sha256?(reference_latent_artifact_sha256) &&
           valid_sha256?(timeline_manifest_sha256) &&
           valid_sha256?(cache_fixture_manifest_sha256)
      raise ArgumentError.new("reference refiner trace identities must be lowercase SHA-256")
    end
    validate_execution!(encode_allocations, encode_readbacks, preterminal_waits, terminal_waits)
    prelude = prelude_metadata(prelude_input_bf16_control,
      prelude_linear_bf16_control, prelude_silu_bf16_control,
      prelude_second_linear_bf16_control, prelude_add_bf16_control)
    unless sources.map(&.name) == TENSOR_ORDER
      raise ArgumentError.new("reference refiner trace source inventory or order mismatch")
    end

    canonical, payload_path = checked_output_paths!(manifest_path)
    ensure_fresh_outputs!(canonical, payload_path)
    pins = compiled_source_sha256s
    records = {} of String => TensorRecord
    payload_digest = Digest::SHA256.new
    offset = 0_i64
    payload_io = HyNomasReferenceCacheProbe.open_fresh_output!(payload_path)
    begin
      sources.each do |source|
        expected_dtype, expected_shape = tensor_spec(source.name, valid_text_tokens)
        unless source.dtype == expected_dtype && source.shape == expected_shape
          raise ArgumentError.new("reference refiner trace source shape/dtype mismatch: #{source.name}")
        end
        expected_bytes = expected_shape.reduce(1_i64) { |count, extent| count * extent } * sizeof(Float32)
        unless source.byte_count.to_i64 == expected_bytes
          raise ArgumentError.new("reference refiner trace source byte count mismatch: #{source.name}")
        end
        bytes = bytes_for(source)
        digest = Digest::SHA256.hexdigest(bytes)
        payload_io.write(bytes)
        payload_digest.update(bytes)
        records[source.name] = TensorRecord.new(expected_dtype, expected_shape, offset,
          expected_bytes, digest, source.checkpoint, source.alias_of)
        offset += expected_bytes
      end
    ensure
      payload_io.close
    end

    payload_sha256 = payload_digest.hexfinal
    payload_name = File.basename(payload_path)
    raw = JSON.build(indent: "  ") do |json|
      json.object do
        json.field("schema", SCHEMA)
        json.field("schema_version", 1)
        json.field("upstream_revision", UPSTREAM_REVISION)
        json.field("scope", SCOPE)
        json.field("full_upstream_t0_preprocessing_parity", false)
        json.field("full_text_refiner_encode_executed", true)
        json.field("capture_boundary", "preblock")
        json.field("prelude_input_bf16_control", prelude[:prelude_input_bf16_control])
        json.field("prelude_linear_bf16_control", prelude[:prelude_linear_bf16_control])
        json.field("prelude_silu_bf16_control", prelude[:prelude_silu_bf16_control])
        json.field("prelude_second_linear_bf16_control", prelude[:prelude_second_linear_bf16_control])
        json.field("prelude_add_bf16_control", prelude[:prelude_add_bf16_control])
        json.field("full_bf16_prelude_parity", prelude[:full_bf16_prelude_parity])
        json.field("prelude_input_bf16_control_scope", prelude[:prelude_input_bf16_control_scope])
        json.field("prelude_linear_bf16_control_scope", prelude[:prelude_linear_bf16_control_scope])
        json.field("prelude_silu_bf16_control_scope", prelude[:prelude_silu_bf16_control_scope])
        json.field("prelude_second_linear_bf16_control_scope", prelude[:prelude_second_linear_bf16_control_scope])
        json.field("prelude_add_bf16_control_scope", prelude[:prelude_add_bf16_control_scope])
        json.field("device", "mps")
        json.field("producer_dtype", "F32")
        json.field("reference_latent_artifact_sha256", reference_latent_artifact_sha256)
        json.field("timeline_manifest_sha256", timeline_manifest_sha256)
        json.field("cache_fixture_manifest_sha256", cache_fixture_manifest_sha256)
        json.field("valid_text_tokens", valid_text_tokens)
        json.field("compiled_source_sha256") do
          json.object { pins.each { |name, digest| json.field(name, digest) } }
        end
        json.field("tensor_order", TENSOR_ORDER)
        json.field("tensor_artifact") do
          json.object do
            json.field("basename", payload_name)
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
                  if alias_of = record.alias_of
                    json.field("alias_of", alias_of)
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
