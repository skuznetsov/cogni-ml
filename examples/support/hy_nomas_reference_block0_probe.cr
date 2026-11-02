require "digest/sha256"
require "json"
require "set"
require "../../src/ml/core/buffer"
require "../../src/ml/video/metal_ops"
require "./hy_nomas_reference_cache_probe"

module HyNomasReferenceBlock0Probe
  extend self

  SCHEMA                      = "cogni_ml.hy_nomas_reference_block0_probe/v1"
  PREQKV_SCHEMA               = "cogni_ml.hy_nomas_reference_block0_preqkv_probe/v1"
  REFERENCE_HISTORY_PRECISION = "reference-bf16-history"
  PREQKV_SCOPE                = "diagnostic pre-QKV boundaries from one standalone native block-0 forward using authenticated native preblock inputs; not a complete upstream T0 handoff"
  PREATTENTION_SCOPE          = "one standalone native block-0 pre-attention forward using native-exported T0 preblock inputs; not a complete upstream T0 handoff"
  PREQKV_PREATTENTION_SCOPE   = "diagnostic pre-QKV boundaries from one standalone native block-0 pre-attention forward using authenticated native preblock inputs; not a complete upstream T0 handoff"
  REFERENCE_BF16_PRECISION    = "reference-bf16-preattention"
  REFERENCE_BF16_VARIANTS     = ["bf16-graph", "f32-accum"]
  NATIVE_REVISION             = "a31c0fe0bcc493be188fd37ddba4ce2f674becb4"
  SOURCE_REVISION             = "911cea6458a89c2663513ff7099561ec190dd491"
  BLOCK0_WEIGHT_SHA256        = "01be7388fa4267f1f566c3ced405418f94d0ed2b4ee969cc880fb2518ab7c87b"
  INPUT_PREFIX_TENSORS        = ["block.img_f32", "block.txt_valid_f32", "block.vec_f32",
                                 "block.vec_txt_f32", "block.rope_cos_f32", "block.rope_sin_f32"]
  SOURCE_CAPTURE_NAMES = [
    "img.q_projection", "img.k_projection", "img.v_projection",
    "img.q_qknorm", "img.k_qknorm", "img.q_rope", "img.k_rope",
    "txt.q_projection", "txt.k_projection", "txt.v_projection",
    "txt.q_qknorm", "txt.k_qknorm", "cache.k_native_f32", "cache.v_native_f32",
  ]
  PREQKV_CAPTURE_NAMES = [
    "img.mod_silu", "img.mod_params", "img.norm1", "img.qkv_input",
    "txt.mod_silu", "txt.mod_params", "txt.norm1", "txt.qkv_input",
  ]
  BATCHES             =     1
  IMAGE_TOKENS        = 4_320
  TEXT_STORAGE_TOKENS = 1_280
  DIM                 = 2_048
  HEADS               =    16
  HEAD_DIM            =   128
  MAX_NAMED_BYTES     = 4_i64 * 1024 * 1024 * 1024

  record TensorRecord, dtype : String, shape : Array(Int32), offset : Int64,
    nbytes : Int64, sha256 : String

  record CaptureTensor, name : String, buffer : ML::MetalBuffer,
    shape : Array(Int32), byte_count : Int32, semantic : String

  record CaptureHeaderEntry, name : String, shape : Array(Int32), byte_count : Int32

  record ManifestInfo, manifest_sha256 : String, payload_name : String,
    payload_bytes : Int64, payload_sha256 : String,
    reference_latent_artifact_sha256 : String, timeline_manifest_sha256 : String,
    valid_text_tokens : Int32, tensors : Hash(String, TensorRecord)

  def valid_sha256?(value : String) : Bool
    /\A[0-9a-f]{64}\z/.matches?(value)
  end

  def validate_preqkv_options!(capture_preqkv : Bool, execute : Bool,
                               artifact_path : String,
                               receipt_path : String) : Nil
    has_artifact = !artifact_path.empty?
    has_receipt = !receipt_path.empty?
    if has_artifact != has_receipt
      raise ArgumentError.new("pre-QKV capture requires both companion output paths")
    end
    if capture_preqkv
      if execute && !has_artifact
        raise ArgumentError.new("--execute --capture-preqkv requires both companion output paths")
      end
    elsif has_artifact || has_receipt
      raise ArgumentError.new("pre-QKV output paths require explicit --capture-preqkv")
    end
  end

  def validate_precision_options!(precision_name : String,
                                  bf16_linear_variant_name : String,
                                  bf16_linear_variant_supplied : Bool) : Nil
    unless {"f32-control", "bf16-boundary-control", REFERENCE_BF16_PRECISION,
            REFERENCE_HISTORY_PRECISION}.includes?(precision_name)
      raise ArgumentError.new("invalid reference block precision")
    end
    if precision_name == REFERENCE_HISTORY_PRECISION
      unless bf16_linear_variant_name == "bf16-graph"
        raise ArgumentError.new("reference-bf16-history requires the bf16-graph linear variant")
      end
    elsif precision_name == REFERENCE_BF16_PRECISION
      unless REFERENCE_BF16_VARIANTS.includes?(bf16_linear_variant_name)
        raise ArgumentError.new("BF16 linear variant must be bf16-graph or f32-accum")
      end
    elsif bf16_linear_variant_supplied
      raise ArgumentError.new("--bf16-linear-variant is only valid with --precision #{REFERENCE_BF16_PRECISION}")
    end
  end

  def capture_shape(name : String, valid_text_tokens : Int32) : Array(Int32)
    unless 0 < valid_text_tokens <= TEXT_STORAGE_TOKENS
      raise ArgumentError.new("reference block valid text prefix must contain 1..1280 tokens")
    end
    case name
    when "img.q_projection", "img.k_projection", "img.v_projection",
         "img.q_qknorm", "img.k_qknorm", "img.q_rope", "img.k_rope"
      [BATCHES, IMAGE_TOKENS, HEADS, HEAD_DIM]
    when "txt.q_projection", "txt.k_projection", "txt.v_projection",
         "txt.q_qknorm", "txt.k_qknorm"
      [BATCHES, valid_text_tokens, HEADS, HEAD_DIM]
    when "cache.k_native_f32", "cache.v_native_f32"
      [BATCHES, IMAGE_TOKENS + valid_text_tokens, HEADS, HEAD_DIM]
    else
      raise ArgumentError.new("unknown reference block capture tensor #{name}")
    end
  end

  def capture_byte_count(name : String, valid_text_tokens : Int32) : Int32
    count = capture_shape(name, valid_text_tokens).reduce(1_i64) do |n, extent|
      n * extent
    end * sizeof(Float32)
    raise ArgumentError.new("reference block capture exceeds Int32 storage") if count > Int32::MAX
    count.to_i32
  end

  def capture_header_entries(valid_text_tokens : Int32) : Array(CaptureHeaderEntry)
    SOURCE_CAPTURE_NAMES.map do |name|
      shape = capture_shape(name, valid_text_tokens)
      CaptureHeaderEntry.new(name, shape, capture_byte_count(name, valid_text_tokens))
    end
  end

  def preqkv_capture_shape(name : String, valid_text_tokens : Int32) : Array(Int32)
    unless 0 < valid_text_tokens <= TEXT_STORAGE_TOKENS
      raise ArgumentError.new("reference pre-QKV valid text prefix must contain 1..1280 tokens")
    end
    case name
    when "img.mod_silu", "txt.mod_silu"
      [BATCHES, DIM]
    when "img.mod_params", "txt.mod_params"
      [BATCHES, 6 * DIM]
    when "img.norm1", "img.qkv_input"
      [BATCHES, IMAGE_TOKENS, DIM]
    when "txt.norm1", "txt.qkv_input"
      [BATCHES, valid_text_tokens, DIM]
    else
      raise ArgumentError.new("unknown reference pre-QKV capture tensor #{name}")
    end
  end

  def preqkv_capture_byte_count(name : String, valid_text_tokens : Int32) : Int32
    count = preqkv_capture_shape(name, valid_text_tokens).reduce(1_i64) do |n, extent|
      n * extent
    end * sizeof(Float32)
    raise ArgumentError.new("reference pre-QKV capture exceeds Int32 storage") if count > Int32::MAX
    count.to_i32
  end

  def preqkv_header_entries(valid_text_tokens : Int32) : Array(CaptureHeaderEntry)
    PREQKV_CAPTURE_NAMES.map do |name|
      shape = preqkv_capture_shape(name, valid_text_tokens)
      CaptureHeaderEntry.new(name, shape, preqkv_capture_byte_count(name, valid_text_tokens))
    end
  end

  def preqkv_precision_label(precision_name : String) : String
    case precision_name
    when "f32-control"            then "F32Control"
    when "bf16-boundary-control"  then "BF16BoundaryControl"
    when REFERENCE_BF16_PRECISION then REFERENCE_BF16_PRECISION
    else
      raise ArgumentError.new("invalid reference pre-QKV precision control")
    end
  end

  def preqkv_metadata_fields(precision_name : String,
                             parent_native_receipt_sha256 : String,
                             parent_native_artifact_sha256 : String,
                             bf16_linear_variant : String? = nil) : Hash(String, String)
    unless valid_sha256?(parent_native_receipt_sha256) && valid_sha256?(parent_native_artifact_sha256)
      raise ArgumentError.new("pre-QKV parent identities must be lowercase SHA-256")
    end
    reference_mode = precision_name == REFERENCE_BF16_PRECISION
    if reference_mode != !bf16_linear_variant.nil?
      raise ArgumentError.new("reference BF16 pre-QKV metadata requires its linear variant")
    end
    if variant = bf16_linear_variant
      unless REFERENCE_BF16_VARIANTS.includes?(variant)
        raise ArgumentError.new("invalid reference BF16 linear variant")
      end
    end
    metadata = {
      "format"                        => PREQKV_SCHEMA,
      "native_revision"               => NATIVE_REVISION,
      "source_revision"               => SOURCE_REVISION,
      "precision_control"             => preqkv_precision_label(precision_name),
      "parent_native_receipt_sha256"  => parent_native_receipt_sha256,
      "parent_native_artifact_sha256" => parent_native_artifact_sha256,
    }
    if variant = bf16_linear_variant
      metadata["bf16_linear_variant"] = variant
    end
    metadata
  end

  def capture_header(entries : Array(CaptureHeaderEntry),
                     expected_names : Array(String) = SOURCE_CAPTURE_NAMES,
                     metadata_fields : Hash(String, String)? = nil) : String
    unless entries.map(&.name) == expected_names
      raise ArgumentError.new("reference block capture header inventory/order mismatch")
    end
    offset = 0_i64
    header = JSON.build do |json|
      json.object do
        if metadata = metadata_fields
          json.field("__metadata__") do
            json.object do
              metadata.each { |name, value| json.field(name, value) }
            end
          end
        end
        entries.each do |entry|
          finish = offset + entry.byte_count
          json.field(entry.name) do
            json.object do
              json.field("dtype", "F32")
              json.field("shape", entry.shape)
              json.field("data_offsets", [offset, finish])
            end
          end
          offset = finish
        end
      end
    end
    header + (" " * ((8 - header.bytesize % 8) % 8))
  end

  # Safetensors stores an 8-byte little-endian header-length prefix, then the
  # padded JSON header, then the concatenated tensor payload.
  def safetensors_file_bytes(header : String, payload_bytes : Int64) : Int64
    raise ArgumentError.new("safetensors payload extent must be nonnegative") if payload_bytes < 0
    prefix_bytes = sizeof(UInt64).to_i64
    header_bytes = header.bytesize.to_i64
    if header_bytes > Int64::MAX - prefix_bytes || payload_bytes > Int64::MAX - prefix_bytes - header_bytes
      raise ArgumentError.new("safetensors file extent overflows Int64")
    end
    prefix_bytes + header_bytes + payload_bytes
  end

  def tensor_spec(name : String, valid_text_tokens : Int32) : Tuple(String, Array(Int32))
    case name
    when "reference.latent_bf16"  then {"BF16", [1, 32, 30, 9, 16]}
    when "cache.noisy_latent_f32" then {"F32", [1, 32, 30, 9, 16]}
    when "block.img_f32"          then {"F32", [1, IMAGE_TOKENS, DIM]}
    when "block.txt_valid_f32"
      unless 0 < valid_text_tokens <= TEXT_STORAGE_TOKENS
        raise ArgumentError.new("reference block text prefix must contain 1..1280 tokens")
      end
      {"F32", [1, valid_text_tokens, DIM]}
    when "block.vec_f32", "block.vec_txt_f32"       then {"F32", [1, DIM]}
    when "block.rope_cos_f32", "block.rope_sin_f32" then {"F32", [IMAGE_TOKENS, HEAD_DIM]}
    when "cache.timestep_f32"                       then {"F32", [1]}
    else                                                 raise ArgumentError.new("unknown reference block input tensor #{name}")
    end
  end

  private def exact_keys!(actual : Hash(String, JSON::Any), expected : Array(String), label : String) : Nil
    unless actual.size == expected.size && actual.keys.to_set == expected.to_set
      raise ArgumentError.new("#{label} key set does not match the reference block contract")
    end
  end

  private def string_field!(object : Hash(String, JSON::Any), name : String) : String
    object[name]?.try(&.as_s) || raise ArgumentError.new("missing or invalid #{name}")
  rescue TypeCastError
    raise ArgumentError.new("missing or invalid #{name}")
  end

  private def integer_field!(object : Hash(String, JSON::Any), name : String) : Int64
    object[name]?.try(&.as_i64) || raise ArgumentError.new("missing or invalid #{name}")
  rescue TypeCastError
    raise ArgumentError.new("missing or invalid #{name}")
  end

  private def object_field!(object : Hash(String, JSON::Any), name : String) : Hash(String, JSON::Any)
    object[name]?.try(&.as_h) || raise ArgumentError.new("missing or invalid #{name}")
  rescue TypeCastError
    raise ArgumentError.new("missing or invalid #{name}")
  end

  private def string_array_field!(object : Hash(String, JSON::Any), name : String) : Array(String)
    object[name]?.try(&.as_a.map(&.as_s)) || raise ArgumentError.new("missing or invalid #{name}")
  rescue TypeCastError
    raise ArgumentError.new("missing or invalid #{name}")
  end

  def validate_manifest_document!(document : JSON::Any,
                                  expected_manifest_sha256 : String) : ManifestInfo
    unless valid_sha256?(expected_manifest_sha256)
      raise ArgumentError.new("expected reference cache manifest identity must be lowercase SHA-256")
    end
    root = document.as_h
    exact_keys!(root, [
      "schema", "schema_version", "upstream_revision", "scope", "preprocessing",
      "reference_latent_artifact_sha256", "timeline_manifest_sha256", "valid_text_tokens",
      "cache_writer", "payload", "payload_bytes", "payload_sha256", "tensors",
    ], "reference cache manifest")
    unless string_field!(root, "schema") == HyNomasReferenceCacheProbe::SCHEMA &&
           integer_field!(root, "schema_version") == 1 &&
           string_field!(root, "upstream_revision") == HyNomasReferenceCacheProbe::UPSTREAM_REVISION &&
           string_field!(root, "scope") == HyNomasReferenceCacheProbe::SCOPE
      raise ArgumentError.new("reference cache manifest identity or scope mismatch")
    end

    preprocessing = object_field!(root, "preprocessing")
    exact_keys!(preprocessing, ["producer", "stages", "full_upstream_t0_preprocessing_parity",
                                "timeline_upstream_revision"], "reference cache preprocessing")
    unless string_field!(preprocessing, "producer") == "native" &&
           string_array_field!(preprocessing, "stages") == ["timeline_reference_scatter", "timeline_cache_noise",
                                                            "patch_projection", "text_refiner"] &&
           preprocessing["full_upstream_t0_preprocessing_parity"]?.try(&.as_bool) == false &&
           string_field!(preprocessing, "timeline_upstream_revision") == SOURCE_REVISION
      raise ArgumentError.new("reference cache preprocessing claim mismatch")
    end

    cache_writer = object_field!(root, "cache_writer")
    exact_keys!(cache_writer, ["intent", "technical_label", "segment_index", "denoising_steps", "sigma", "timestep"],
      "reference cache writer")
    unless string_field!(cache_writer, "intent") == "single_reference_initialization_forward" &&
           string_field!(cache_writer, "technical_label") == "T0" &&
           integer_field!(cache_writer, "segment_index") == 0 &&
           integer_field!(cache_writer, "denoising_steps") == 0 &&
           cache_writer["sigma"]?.try(&.as_f) == 0.206897 &&
           cache_writer["timestep"]?.try(&.as_f) == 206.897
      raise ArgumentError.new("reference cache writer contract mismatch")
    end

    valid_tokens = integer_field!(root, "valid_text_tokens")
    unless valid_tokens > 0 && valid_tokens <= TEXT_STORAGE_TOKENS
      raise ArgumentError.new("reference block valid text prefix must contain 1..1280 tokens")
    end
    latent_sha = string_field!(root, "reference_latent_artifact_sha256")
    timeline_sha = string_field!(root, "timeline_manifest_sha256")
    payload_sha = string_field!(root, "payload_sha256")
    unless valid_sha256?(latent_sha) && valid_sha256?(timeline_sha) && valid_sha256?(payload_sha)
      raise ArgumentError.new("reference cache input identities must be lowercase SHA-256")
    end

    payload_name = string_field!(root, "payload")
    unless !payload_name.empty? && File.basename(payload_name) == payload_name &&
           payload_name != "." && payload_name != ".." &&
           !payload_name.includes?('/') && !payload_name.includes?('\\')
      raise ArgumentError.new("reference cache payload must be a sibling basename")
    end
    declared_payload_bytes = integer_field!(root, "payload_bytes")
    tensor_objects = object_field!(root, "tensors")
    exact_keys!(tensor_objects, HyNomasReferenceCacheProbe::TENSOR_NAMES, "reference cache tensor inventory")
    tensors = {} of String => TensorRecord
    offset = 0_i64
    HyNomasReferenceCacheProbe::TENSOR_NAMES.each do |name|
      raw = object_field!(tensor_objects, name)
      exact_keys!(raw, ["dtype", "shape", "offset", "nbytes", "sha256"], "reference tensor #{name}")
      dtype, shape = tensor_spec(name, valid_tokens.to_i32)
      shape_value = raw["shape"]?.try(&.as_a.map(&.as_i.to_i32)) ||
                    raise ArgumentError.new("reference tensor shape is missing: #{name}")
      nbytes = integer_field!(raw, "nbytes")
      record_offset = integer_field!(raw, "offset")
      tensor_sha = string_field!(raw, "sha256")
      expected_nbytes = shape.reduce(1_i64) { |count, extent| count * extent } * (dtype == "BF16" ? 2 : 4)
      unless string_field!(raw, "dtype") == dtype && shape_value == shape &&
             nbytes == expected_nbytes && record_offset == offset && valid_sha256?(tensor_sha)
        raise ArgumentError.new("reference cache tensor contract mismatch: #{name}")
      end
      if nbytes > Int64::MAX - offset
        raise ArgumentError.new("reference cache tensor extent overflow")
      end
      tensors[name] = TensorRecord.new(dtype, shape, offset, nbytes, tensor_sha)
      offset += nbytes
    end
    unless declared_payload_bytes == offset
      raise ArgumentError.new("reference cache payload length differs from tensor extents")
    end

    ManifestInfo.new(expected_manifest_sha256, payload_name, declared_payload_bytes,
      payload_sha, latent_sha, timeline_sha, valid_tokens.to_i32, tensors)
  rescue TypeCastError
    raise ArgumentError.new("reference cache manifest has an invalid field type")
  end

  class InputBundle
    getter info : ManifestInfo
    getter manifest_path : String
    getter payload_path : String
    @payload : File
    @payload_initial_info : File::Info
    @closed = false

    def initialize(path : String, expected_manifest_sha256 : String)
      unless Path[path].absolute? && File.extname(path) == ".json"
        raise ArgumentError.new("reference block input manifest must be an absolute JSON path")
      end
      raise ArgumentError.new("reference block input manifest must not be a symlink") if File.symlink?(path)
      @manifest_path = File.join(File.realpath(File.dirname(path)), File.basename(path))
      manifest_io = File.open(@manifest_path, "rb")
      manifest_bytes = begin
        unless File.file?(@manifest_path) && manifest_io.info.size <= 1024 * 1024
          raise ArgumentError.new("reference block input manifest is not a bounded regular file")
        end
        manifest_io.gets_to_end
      ensure
        manifest_io.close
      end
      unless HyNomasReferenceBlock0Probe.valid_sha256?(expected_manifest_sha256) &&
             Digest::SHA256.hexdigest(manifest_bytes) == expected_manifest_sha256
        raise ArgumentError.new("reference block input manifest SHA-256 mismatch")
      end
      @info = HyNomasReferenceBlock0Probe.validate_manifest_document!(JSON.parse(manifest_bytes),
        expected_manifest_sha256)
      @payload_path = File.join(File.dirname(@manifest_path), @info.payload_name)
      raise ArgumentError.new("reference block payload must not be a symlink") if File.symlink?(@payload_path)
      @payload = File.open(@payload_path, "rb")
      @payload_initial_info = @payload.info
      begin
        unless File.file?(@payload_path) && @payload_initial_info.size == @info.payload_bytes
          raise ArgumentError.new("reference block payload file extent/type mismatch")
        end
        verify_payload!
      rescue ex
        @payload.close
        raise ex
      end
    end

    def verify_payload! : Nil
      raise ArgumentError.new("reference block input bundle is closed") if @closed
      check_payload_identity!
      @payload.rewind
      digest = Digest::SHA256.new
      buffer = Bytes.new(1024 * 1024)
      while (count = @payload.read(buffer)) > 0
        digest.update(buffer[0, count])
      end
      unless digest.hexfinal == @info.payload_sha256
        raise ArgumentError.new("reference block payload SHA-256 mismatch")
      end
      check_payload_identity!
      @payload.rewind
    end

    def upload!(name : String, arena : ML::Video::Arena) : ML::MetalBuffer
      record = @info.tensors[name]? || raise ArgumentError.new("unknown reference block input #{name}")
      unless HyNomasReferenceBlock0Probe::INPUT_PREFIX_TENSORS.includes?(name) && record.dtype == "F32"
        raise ArgumentError.new("reference block input is not an admitted F32 activation: #{name}")
      end
      bytes = read_tensor!(name)
      buffer = arena.alloc(name, (record.nbytes // sizeof(Float32)).to_i32)
      buffer.write_bytes(bytes.to_unsafe, bytes.size)
      buffer
    end

    # The native preblock fixture stores the authenticated valid text prefix
    # only. Extend it with zeros for the production 1280-row storage shape;
    # attention still selects exactly valid_text_tokens rows.
    def upload_text_storage!(arena : ML::Video::Arena) : ML::MetalBuffer
      record = @info.tensors["block.txt_valid_f32"]
      valid_bytes = read_tensor!("block.txt_valid_f32")
      storage_bytes = TEXT_STORAGE_TOKENS.to_i64 * DIM * sizeof(Float32)
      buffer = arena.alloc("block.txt_storage_f32", (storage_bytes // sizeof(Float32)).to_i32)
      zeros = Bytes.new(storage_bytes.to_i32, 0_u8)
      buffer.write_bytes(zeros.to_unsafe, zeros.size)
      buffer.write_bytes(valid_bytes.to_unsafe, valid_bytes.size)
      buffer
    end

    def close : Nil
      return if @closed
      @closed = true
      @payload.close
    end

    private def read_tensor!(name : String) : Bytes
      raise ArgumentError.new("reference block input bundle is closed") if @closed
      unless HyNomasReferenceBlock0Probe::INPUT_PREFIX_TENSORS.includes?(name)
        raise ArgumentError.new("reference block tensor is not an admitted input")
      end
      record = @info.tensors[name]
      check_payload_identity!
      @payload.seek(record.offset)
      bytes = Bytes.new(record.nbytes.to_i32)
      @payload.read_fully(bytes)
      check_payload_identity!
      unless Digest::SHA256.hexdigest(bytes) == record.sha256
        raise ArgumentError.new("reference block tensor SHA-256 mismatch")
      end
      bytes
    end

    private def check_payload_identity! : Nil
      current = @payload.info
      unless current.same_file?(@payload_initial_info) && current.size == @payload_initial_info.size &&
             current.modification_time == @payload_initial_info.modification_time
        raise ArgumentError.new("reference block payload changed after admission")
      end
    end
  end
end
