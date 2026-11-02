require "json"

module HyNomasReferenceHistoryProbe
  extend self

  SCHEMA              = "cogni_ml.hy_nomas_reference_history_probe/v1"
  SCOPE               = "one standalone native block-0 BF16 pre-attention forward with first T0 history store/read; not a complete upstream T0 handoff"
  PRECISION           = "reference-bf16-history"
  LINEAR_VARIANT      = "bf16-graph"
  BATCHES             =     1
  IMAGE_TOKENS        = 4_320
  TEXT_STORAGE_TOKENS = 1_280
  STORAGE_ROWS        = IMAGE_TOKENS + TEXT_STORAGE_TOKENS
  DIM                 = 2_048
  HEADS               =    16
  HEAD_DIM            =   128
  TENSOR_NAMES        = [
    "history.k_bf16", "history.v_bf16", "history.k_read_f32", "history.v_read_f32",
  ]
  LAYOUT = "[1,tokens,heads,head_dim]; image then valid text"

  record TensorEntry, name : String, dtype : String, shape : Array(Int32), byte_count : Int32
  record TensorArtifact, name : String, dtype : String, shape : Array(Int32),
    nbytes : Int32, sha256 : String

  def validate_options!(precision_name : String, bf16_linear_variant_name : String,
                        capture_preqkv : Bool, preqkv_artifact_path : String,
                        preqkv_receipt_path : String) : Nil
    return unless precision_name == PRECISION
    unless bf16_linear_variant_name == LINEAR_VARIANT
      raise ArgumentError.new("reference-bf16-history requires the bf16-graph linear variant")
    end
    if capture_preqkv || !preqkv_artifact_path.empty? || !preqkv_receipt_path.empty?
      raise ArgumentError.new("reference-bf16-history does not support --capture-preqkv or pre-QKV output paths")
    end
  end

  # The source text buffer has 1280 rows of capacity. The physical BF16 cache
  # and F32 read staging are exact-sized to the active image plus valid text
  # prefix; storage_rows is only the geometry bound.
  def geometry(storage_rows : Int32, active_rows : Int32,
               heads : Int32 = HEADS, head_dim : Int32 = HEAD_DIM)
    unless storage_rows > 0 && active_rows > 0 && active_rows <= storage_rows &&
           heads > 0 && head_dim > 0
      raise ArgumentError.new("reference history geometry is outside its storage bounds")
    end
    elements = 1_i64
    {active_rows, heads, head_dim}.each do |dimension|
      if elements > Int32::MAX.to_i64 // dimension
        raise ArgumentError.new("reference history cache element count exceeds the Metal limit")
      end
      elements *= dimension
    end
    persisted_bytes = elements * 4_i64
    read_stage_bytes = elements * 8_i64
    {
      storage_rows:     storage_rows,
      active_rows:      active_rows,
      heads:            heads,
      head_dim:         head_dim,
      cache_elements:   elements,
      persisted_bytes:  persisted_bytes,
      read_stage_bytes: read_stage_bytes,
      named_bytes:      persisted_bytes + read_stage_bytes,
    }
  end

  def named_bytes(storage_rows : Int32, active_rows : Int32,
                  heads : Int32 = HEADS, head_dim : Int32 = HEAD_DIM) : Int64
    geometry(storage_rows, active_rows, heads, head_dim)[:named_bytes]
  end

  def active_rows(valid_text_tokens : Int32) : Int32
    unless 0 < valid_text_tokens <= TEXT_STORAGE_TOKENS
      raise ArgumentError.new("reference history valid text prefix must contain 1..1280 tokens")
    end
    IMAGE_TOKENS + valid_text_tokens
  end

  def tensor_shape(name : String, valid_text_tokens : Int32) : Array(Int32)
    unless TENSOR_NAMES.includes?(name)
      raise ArgumentError.new("unknown reference history tensor #{name}")
    end
    [BATCHES, active_rows(valid_text_tokens), HEADS, HEAD_DIM]
  end

  def tensor_dtype(name : String) : String
    case name
    when "history.k_bf16", "history.v_bf16"         then "BF16"
    when "history.k_read_f32", "history.v_read_f32" then "F32"
    else
      raise ArgumentError.new("unknown reference history tensor #{name}")
    end
  end

  def tensor_byte_count(name : String, valid_text_tokens : Int32) : Int32
    bytes_per_element = tensor_dtype(name) == "BF16" ? 2_i64 : 4_i64
    count = tensor_shape(name, valid_text_tokens).reduce(1_i64) do |n, extent|
      n * extent
    end * bytes_per_element
    raise ArgumentError.new("reference history tensor exceeds Int32 storage") if count > Int32::MAX
    count.to_i32
  end

  def header_entries(valid_text_tokens : Int32) : Array(TensorEntry)
    TENSOR_NAMES.map do |name|
      TensorEntry.new(name, tensor_dtype(name), tensor_shape(name, valid_text_tokens),
        tensor_byte_count(name, valid_text_tokens))
    end
  end

  def safetensors_header(entries : Array(TensorEntry)) : String
    unless entries.map(&.name) == TENSOR_NAMES
      raise ArgumentError.new("reference history safetensors inventory/order mismatch")
    end
    unless entries.all? { |entry| entry.shape.size == 4 }
      raise ArgumentError.new("reference history safetensors tensor rank mismatch")
    end
    expected_entries = header_entries(entries.first.shape[1] - IMAGE_TOKENS)
    entries.zip(expected_entries).each do |entry, expected|
      unless entry.name == expected.name && entry.dtype == expected.dtype &&
             entry.shape == expected.shape && entry.byte_count == expected.byte_count
        raise ArgumentError.new("reference history safetensors tensor geometry mismatch: #{entry.name}")
      end
    end
    offset = 0_i64
    header = JSON.build do |json|
      json.object do
        entries.each do |entry|
          unless {"BF16", "F32"}.includes?(entry.dtype) && entry.byte_count > 0
            raise ArgumentError.new("invalid reference history tensor header entry: #{entry.name}")
          end
          finish = offset + entry.byte_count
          json.field(entry.name) do
            json.object do
              json.field("dtype", entry.dtype)
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

  def safetensors_file_bytes(header : String, payload_bytes : Int64) : Int64
    raise ArgumentError.new("safetensors payload extent must be nonnegative") if payload_bytes < 0
    prefix_bytes = sizeof(UInt64).to_i64
    header_bytes = header.bytesize.to_i64
    if header_bytes > Int64::MAX - prefix_bytes || payload_bytes > Int64::MAX - prefix_bytes - header_bytes
      raise ArgumentError.new("safetensors file extent overflows Int64")
    end
    prefix_bytes + header_bytes + payload_bytes
  end

  def receipt_json(native_revision : String, source_revision : String,
                   input_manifest_sha256 : String, input_payload_sha256 : String,
                   reference_latent_sha256 : String, timeline_manifest_sha256 : String,
                   valid_text_tokens : Int32, weights_sha256 : String,
                   artifact_basename : String, artifact_sha256 : String,
                   artifact_bytes : Int64, records : Array(TensorArtifact),
                   named_buffer_accounting : Hash(String, Int64),
                   encode_allocations : Int64, encode_readbacks : Int64,
                   preterminal_waits : Int32, terminal_waits : Int32) : String
    identities = [input_manifest_sha256, input_payload_sha256,
                  reference_latent_sha256, timeline_manifest_sha256,
                  weights_sha256, artifact_sha256]
    unless valid_revision?(native_revision) && valid_revision?(source_revision) &&
           identities.all? { |identity| valid_sha256?(identity) }
      raise ArgumentError.new("reference history receipt identities must be lowercase SHA-256")
    end
    unless !artifact_basename.empty? && File.basename(artifact_basename) == artifact_basename &&
           artifact_basename.ends_with?(".safetensors") && artifact_bytes > 0
      raise ArgumentError.new("reference history artifact receipt fields are invalid")
    end
    active = active_rows(valid_text_tokens)
    expected = header_entries(valid_text_tokens)
    unless records.map(&.name) == TENSOR_NAMES
      raise ArgumentError.new("reference history receipt tensor inventory/order mismatch")
    end
    records.zip(expected).each do |record, entry|
      unless record.name == entry.name && record.dtype == entry.dtype &&
             record.shape == entry.shape && record.nbytes == entry.byte_count &&
             valid_sha256?(record.sha256)
        raise ArgumentError.new("reference history receipt tensor contract mismatch: #{record.name}")
      end
    end
    expected_file_bytes = safetensors_file_bytes(
      safetensors_header(expected), expected.sum(0_i64) { |entry| entry.byte_count.to_i64 })
    unless artifact_bytes == expected_file_bytes
      raise ArgumentError.new("reference history artifact byte count does not match its tensor geometry")
    end
    unless encode_allocations == 0 && encode_readbacks == 0 &&
           preterminal_waits == 0 && terminal_waits == 1
      raise ArgumentError.new("reference history receipt requires zero encode allocations/readbacks/waits and one terminal wait")
    end
    dimensions = geometry(STORAGE_ROWS, active)
    accounting_parts = [
      "input_buffers", "block_weight_residents", "block_scratch", "mps_weight_stage",
      "compact_attention_buffers", "qk_rms_square_and_mean_buffers",
      "reference_bf16_buffers", "reference_history_buffers",
    ]
    expected_accounting_names = accounting_parts + ["total_named_bytes"]
    unless named_buffer_accounting.size == expected_accounting_names.size &&
           expected_accounting_names.all? { |name| named_buffer_accounting.has_key?(name) } &&
           named_buffer_accounting["reference_history_buffers"] == dimensions[:named_bytes]
      raise ArgumentError.new("reference history named-buffer accounting is incomplete")
    end
    accounting_total = accounting_parts.sum(0_i64) do |name|
      value = named_buffer_accounting[name]
      raise ArgumentError.new("reference history named-buffer sizes must be nonnegative") if value < 0
      value
    end
    unless named_buffer_accounting["total_named_bytes"] == accounting_total
      raise ArgumentError.new("reference history named-buffer total does not match its parts")
    end

    JSON.build(indent: "  ") do |json|
      json.object do
        json.field("schema", SCHEMA)
        json.field("schema_version", 1)
        json.field("native_revision", native_revision)
        json.field("source_revision", source_revision)
        json.field("scope", SCOPE)
        json.field("full_upstream_t0_preprocessing_parity", false)
        json.field("history_cache_write_executed", true)
        json.field("history_cache_read_executed", true)
        json.field("attention_executed", false)
        json.field("input_bundle_manifest_sha256", input_manifest_sha256)
        json.field("input_bundle_payload_sha256", input_payload_sha256)
        json.field("reference_latent_artifact_sha256", reference_latent_sha256)
        json.field("timeline_manifest_sha256", timeline_manifest_sha256)
        json.field("valid_text_tokens", valid_text_tokens)
        json.field("text_storage_tokens", TEXT_STORAGE_TOKENS)
        json.field("weights_sha256", weights_sha256)
        json.field("precision_control", PRECISION)
        json.field("bf16_linear_variant", LINEAR_VARIANT)
        json.field("history") do
          json.object do
            json.field("storage_dtype", "BF16")
            json.field("read_dtype", "F32")
            json.field("layout", LAYOUT)
            json.field("cache_valid_len", 0)
            json.field("cache_txt", true)
            json.field("stored_tokens", active)
            json.field("cache_writes", 1)
            json.field("cache_reads", 1)
            json.field("persisted_bytes", dimensions[:persisted_bytes])
            json.field("read_stage_bytes", dimensions[:read_stage_bytes])
          end
        end
        json.field("tensor_artifact") do
          json.object do
            json.field("basename", artifact_basename)
            json.field("sha256", artifact_sha256)
            json.field("bytes", artifact_bytes)
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
        json.field("named_buffer_accounting") do
          json.object do
            named_buffer_accounting.each { |name, bytes| json.field(name, bytes) }
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
  end

  private def valid_sha256?(value : String) : Bool
    /\A[0-9a-f]{64}\z/.matches?(value)
  end

  private def valid_revision?(value : String) : Bool
    /\A[0-9a-f]{40}\z/.matches?(value)
  end
end
