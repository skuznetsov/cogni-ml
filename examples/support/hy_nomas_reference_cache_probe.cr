require "digest/sha256"
require "json"
require "../../src/ml/core/buffer"

module HyNomasReferenceCacheProbe
  extend self

  SCHEMA            = "cogni_ml.hy_nomas_reference_cache_inputs/v1"
  UPSTREAM_REVISION = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
  SCOPE             = "native T0 timeline/noise/patch/refiner inputs for a standalone block-0 cache writer; not a complete upstream T0 handoff"
  PREPROCESSING     = {
    "producer"                              => "native",
    "stages"                                => ["timeline_reference_scatter", "timeline_cache_noise", "patch_projection", "text_refiner"],
    "full_upstream_t0_preprocessing_parity" => false,
    "timeline_upstream_revision"            => "911cea6458a89c2663513ff7099561ec190dd491",
  }
  CACHE_WRITER = {
    "intent"          => "single_reference_initialization_forward",
    "technical_label" => "T0",
    "segment_index"   => 0,
    "denoising_steps" => 0,
    "sigma"           => 0.206897,
    "timestep"        => 206.897,
  }
  # The order is part of the source-oracle artifact contract. Offsets are
  # contiguous in this order; each record hash covers its exact raw bytes.
  TENSOR_NAMES = [
    "reference.latent_bf16", "cache.noisy_latent_f32", "block.img_f32",
    "block.txt_valid_f32", "block.vec_f32", "block.vec_txt_f32",
    "block.rope_cos_f32", "block.rope_sin_f32", "cache.timestep_f32",
  ]

  def tensor_spec(name : String, valid_text_tokens : Int32) : Tuple(String, Array(Int32))
    case name
    when "reference.latent_bf16"  then {"BF16", [1, 32, 30, 9, 16]}
    when "cache.noisy_latent_f32" then {"F32", [1, 32, 30, 9, 16]}
    when "block.img_f32"          then {"F32", [1, 4320, 2048]}
    when "block.txt_valid_f32"
      unless 0 < valid_text_tokens <= 1280
        raise ArgumentError.new("reference text prefix must contain 1..1280 tokens")
      end
      {"F32", [1, valid_text_tokens, 2048]}
    when "block.vec_f32", "block.vec_txt_f32"       then {"F32", [1, 2048]}
    when "block.rope_cos_f32", "block.rope_sin_f32" then {"F32", [4320, 128]}
    when "cache.timestep_f32"                       then {"F32", [1]}
    else                                                 raise ArgumentError.new("unknown reference cache tensor #{name}")
    end
  end

  record TensorSource, name : String, buffer : ML::MetalBuffer,
    dtype : String, shape : Array(Int32), byte_count : Int32

  record TensorRecord, dtype : String, shape : Array(Int32), offset : Int64,
    nbytes : Int64, sha256 : String

  @@in_flight_owners = [] of Proc(Nil)

  # Keep all resources strongly reachable when a terminal wait fails.
  def retain_in_flight!(owners : Proc(Nil)) : Nil
    @@in_flight_owners << owners
  end

  def valid_sha256?(value : String) : Bool
    /\A[0-9a-f]{64}\z/.matches?(value)
  end

  def checked_output_paths!(manifest_path : String) : Tuple(String, String)
    unless Path[manifest_path].absolute? && manifest_path.ends_with?(".json")
      raise ArgumentError.new("reference cache output must be an absolute .json path")
    end
    parent = File.realpath(File.dirname(manifest_path))
    canonical = File.join(parent, File.basename(manifest_path))
    payload_path = canonical.sub(/\.json\z/, ".tensors.bin")
    {canonical, payload_path}
  end

  def ensure_fresh_outputs!(manifest_path : String, payload_path : String) : Nil
    {manifest_path, payload_path}.each do |path|
      if File.exists?(path) || File.symlink?(path)
        raise ArgumentError.new("refusing to replace reference cache output: #{path}")
      end
    end
  end

  # The output freshness check is advisory; O_EXCL is the actual
  # no-replacement boundary if another process races artifact creation.
  def open_fresh_output!(path : String) : IO::FileDescriptor
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("reference cache output must be fresh and writable: #{path}") if fd < 0
    IO::FileDescriptor.new(fd)
  end

  def bytes_for(source : TensorSource) : Bytes
    # `block.txt_valid_f32` captures only the authenticated valid prefix from
    # the refiner's 1280-token padded output buffer.
    unless source.buffer.valid? && source.buffer.storage_mode.shared? &&
           source.buffer.size >= source.byte_count
      raise ArgumentError.new("reference cache capture buffer extent/storage mismatch: #{source.name}")
    end
    bytes = Bytes.new(source.byte_count)
    source.buffer.read_bytes(bytes.to_unsafe, bytes.size)
    bytes
  end

  # Called only after the producer's sole terminal wait. All activation reads
  # happen here, outside the measured command-chain region.
  def write_fixture!(manifest_path : String, *,
                     reference_latent_artifact_sha256 : String,
                     timeline_manifest_sha256 : String,
                     valid_text_tokens : Int32,
                     sources : Array(TensorSource)) : Hash(String, JSON::Any)
    unless valid_sha256?(reference_latent_artifact_sha256) && valid_sha256?(timeline_manifest_sha256)
      raise ArgumentError.new("reference cache source identities must be lowercase SHA-256")
    end
    expected_names = TENSOR_NAMES
    unless sources.map(&.name) == expected_names
      raise ArgumentError.new("reference cache source inventory or order mismatch")
    end
    canonical, payload_path = checked_output_paths!(manifest_path)
    ensure_fresh_outputs!(canonical, payload_path)

    records = {} of String => TensorRecord
    payload_digest = Digest::SHA256.new
    offset = 0_i64
    payload_io = open_fresh_output!(payload_path)
    begin
      sources.each do |source|
        expected_dtype, expected_shape = tensor_spec(source.name, valid_text_tokens)
        unless source.dtype == expected_dtype && source.shape == expected_shape
          raise ArgumentError.new("reference cache source shape/dtype mismatch: #{source.name}")
        end
        expected_bytes = expected_shape.reduce(1_i64) { |n, extent| n * extent } * (expected_dtype == "BF16" ? 2 : 4)
        unless source.byte_count.to_i64 == expected_bytes
          raise ArgumentError.new("reference cache source byte count mismatch: #{source.name}")
        end
        bytes = bytes_for(source)
        digest = Digest::SHA256.hexdigest(bytes)
        payload_io.write(bytes)
        payload_digest.update(bytes)
        records[source.name] = TensorRecord.new(expected_dtype, expected_shape, offset,
          expected_bytes, digest)
        offset += expected_bytes
      end
    ensure
      payload_io.close
    end

    payload_name = File.basename(payload_path)
    payload_sha256 = payload_digest.hexfinal
    document = {
      "schema"                           => SCHEMA,
      "schema_version"                   => 1,
      "upstream_revision"                => UPSTREAM_REVISION,
      "scope"                            => SCOPE,
      "preprocessing"                    => PREPROCESSING,
      "reference_latent_artifact_sha256" => reference_latent_artifact_sha256,
      "timeline_manifest_sha256"         => timeline_manifest_sha256,
      "valid_text_tokens"                => valid_text_tokens,
      "cache_writer"                     => CACHE_WRITER,
      "payload"                          => payload_name,
      "payload_bytes"                    => offset,
      "payload_sha256"                   => payload_sha256,
      "tensors"                          => records.transform_values do |record|
        {"dtype" => record.dtype, "shape" => record.shape, "offset" => record.offset,
         "nbytes" => record.nbytes, "sha256" => record.sha256}
      end,
    }
    raw = JSON.build(indent: "  ") do |json|
      json.object { document.each { |key, value| json.field(key, value) } }
    end
    manifest_io = open_fresh_output!(canonical)
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
